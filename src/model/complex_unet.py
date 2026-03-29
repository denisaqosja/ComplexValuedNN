import math
import torch
from inspect import isfunction
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from model.modules.complex_layer_norm_methods import ComplexBatchNorm, ComplexGroupNorm

group_norm = True
complex_norm_layer = ComplexGroupNorm if group_norm else ComplexBatchNorm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def softmax_for_complex_data(x):
    """
    Compute the softmax for both magnitude and phase of complex-valued attention scores
    """
    # obtain the magnitude and phase from the dot-product of Q and K
    magnitude, phase = x.abs(), x.angle()
    # calculate the cosine()
    phase_scores = torch.cos(phase)
    softmax_scores = F.softmax(magnitude * phase_scores, dim=-1)

    return softmax_scores


# disclaimer: The xavier_uniform_() initialization works for complex data as well, without throwing any errors
# but according to literature, sicne real and imag are being initialized, it is recommended to divide the var by 2.
def complex_xavier_uniform_(tensor, gain=1.0):
    """
    Complex-valued Xavier (Glorot) uniform initialization.
    Ensures the total variance of the complex weight matches real Xavier.
    
    Args:
        tensor (torch.Tensor): Complex-valued tensor to initialize.
        gain (float): Scaling factor (default = 1.0)
    """
    if not torch.is_complex(tensor):
        raise ValueError("Tensor must be complex dtype")

    # Compute fan_in and fan_out from real part (same shape)
    n_neurons_in, n_neurons_out = init._calculate_fan_in_and_fan_out(tensor.real)
    std = gain * (2.0 / (n_neurons_in + n_neurons_out)) ** 0.5
    bound = (3.0 ** 0.5) * std

    # Adjust for complex domain: divide by √2 to maintain total variance
    bound /= 2 ** 0.5

    with torch.no_grad():
        real = tensor.real.uniform_(-bound, bound)
        imag = tensor.imag.uniform_(-bound, bound)
        tensor.copy_(torch.complex(real, imag))
    return tensor


def complex_he_uniform_(tensor, gain=1.0):
    """
    Complex-valued Xavier (Glorot) uniform initialization.
    The weights might be initialized too high for complex data with He!! 
    That is why it doesn't work as well as Xavier. 
    
    Args:
        tensor (torch.Tensor): Complex-valued tensor to initialize.
        gain (float): Scaling factor (default = 1.0)
    """
    if not torch.is_complex(tensor):
        raise ValueError("Tensor must be complex dtype")

    # Compute fan_in from real part (same shape)
    fan_in = init._calculate_correct_fan(tensor.real, mode="fan_in")
    # the gain for nonlineariry = "relu" is sqrt(2) -> relu_gain = 2.0 **0.5
    std = gain * (2.0 / fan_in) ** 0.5               # std = gain * relu_gain * (1/fan_in)**0.5  
    bound = (3.0 ** 0.5) * std

    # Adjust for complex domain: divide by √2 to maintain total variance
    bound /= 2 ** 0.5

    with torch.no_grad():
        real = tensor.real.uniform_(-bound, bound)
        imag = tensor.imag.uniform_(-bound, bound)
        tensor.copy_(torch.complex(real, imag))
    return tensor


class ComplexDropout(nn.Module):
    """
    Use Bernoulli Distribution to draw binary random numbers (0 or 1) for keep or dropout neurons.
    out_i ~ Bernoulli(p=input_i); the i-th element will draw a value 1 (probability to keep neurons)
    Therefore, give as input to Bernoulli the keep (success) probability: 1 - p. 
    
    Why normalize during training? During training, the network has fewer neurons. Therefore, the expected value of 
    the output neuron is lower for training than it is for inference. 
    With the scaling factor = 1/(1-p), we make the expected value
        (y = sum_i(x_i * w_i)) 
    same for training and inference.
    """
    
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p 

    def forward(self, x):
        # if inference or if p=0, then ignore dropout and return simply x
        if not self.training or self.p == 0.0:
            return x
        # Bernoulli mask (real, same shape as x). 
        mask = torch.empty_like(x.real).bernoulli_(1 - self.p) / (1 - self.p)
        # Multiply mask with complex tensor
        return x * mask
       
    
class ModReLU(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        modReLU activation function.
        
        Args:
            num_features: number of channels or units (size of bias parameter b)
            eps: small constant to avoid division by zero
        """
        super().__init__()
        # trainable bias
        self.b = nn.Parameter(torch.zeros(num_features))  
        self.eps = eps

    def forward(self, z):
        # Complex z-input (torch.complex64 or torch.complex128)
        magnitude = torch.abs(z) + self.eps
        phase = z / magnitude
        bias = self.b.view(1, -1, *([1] * (z.ndim - 2)))
        activated = torch.relu(magnitude + bias) * phase
        return activated
    

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1, dtype=torch.cfloat)
        self.initialize()

    def initialize(self):
        complex_xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias.real)
        init.zeros_(self.main.bias.imag)
        return 
    
    def forward(self, x):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, dtype=torch.cfloat)
        self.initialize()

    def initialize(self):
        complex_xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias.real)
        init.zeros_(self.main.bias.imag)
        return 

    def forward(self, x):
        _, _, H, W = x.shape
        # since the interpolation is nearest neighbor NN (repeating values), 
        # the real and imag parts can be repeated separately 
        x_real = F.interpolate(x.real, scale_factor=2, mode='nearest')
        x_imag = F.interpolate(x.imag, scale_factor=2, mode='nearest')
        # after the NN on real and imag parts separated, join them into complex numbers
        x = torch.complex(x_real, x_imag)
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.complex_batch_norm = complex_norm_layer(in_ch) 
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, dtype=torch.cfloat)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, dtype=torch.cfloat)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, dtype=torch.cfloat)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, dtype=torch.cfloat)
        self.initialize()


    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            complex_xavier_uniform_(module.weight)
            init.zeros_(module.bias.real)
            init.zeros_(module.bias.imag)
        complex_xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.complex_batch_norm(x)  # h = x 
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        # for complex matrices, the dot product between Q and K is: Q K_hermitian (implemented: k.mH) -> conjugate transpose 
        # since the matrices are already transposed by the .permute(), here we take only the .conjugate
        w = torch.bmm(q, k.conj()) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
     
        w = softmax_for_complex_data(w)   # the output of softmax should be real-valued weights 
        
        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        if v.dtype == torch.complex64:
            # enforce w to be complex
            w = w.to(torch.complex64)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)

        h = self.proj(h)
        self.attn_weights = w

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            complex_norm_layer(in_ch), 
            ModReLU(in_ch), 
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, dtype=torch.cfloat),
        )
    
        self.block2 = nn.Sequential(
            complex_norm_layer(out_ch), 
            ModReLU(out_ch),
            ComplexDropout(dropout),        
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, dtype=torch.cfloat),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, dtype=torch.cfloat)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                complex_xavier_uniform_(module.weight)
                init.zeros_(module.bias.real)
                init.zeros_(module.bias.imag)
            if isinstance(module, nn.Linear):  
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        complex_xavier_uniform_(self.block2[-1].weight, gain=1e-5)
        return 

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, inner_channel=64, channel_mults=[1, 2, 3, 4], attn_res=[2], num_res_blocks=2, dropout=0.2):
        super().__init__()
        assert all([i < len(channel_mults) for i in attn_res]), 'attn index out of bound'
        tdim = inner_channel * 4

        self.head = nn.Conv2d(in_channel, inner_channel, kernel_size=3, stride=1, padding=1, dtype=torch.cfloat)
        self.downblocks = nn.ModuleList()
        chs = [inner_channel]  # record output channel when dowmsample for upsample
        now_ch = inner_channel
        for i, mult in enumerate(channel_mults):
            out_ch = inner_channel * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn_res)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(channel_mults) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = inner_channel * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn_res)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            complex_norm_layer(now_ch),
            ModReLU(now_ch), 
            nn.Conv2d(now_ch, out_channel, kernel_size=3, stride=1, padding=1, dtype=torch.cfloat)
        )
        self.initialize()

    def initialize(self):
        complex_xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias.real)
        init.zeros_(self.head.bias.imag)

        complex_xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias.real)
        init.zeros_(self.tail[-1].bias.imag)
        return 


    def forward(self, x):
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h)
        h = self.tail(h)

        assert len(hs) == 0
        return h


# if __name__ == "__main__":
#     T = 10
#     # device = torch.device("cuda" if torch.cuda.is_available else "cpu")
#     device = "cpu"
#     print(device)
#     x = torch.randn(size=(4, 2, 128, 128), dtype=torch.cfloat)
#     x = x.to(device)
#     t = torch.randint(T, size=(x.shape[0], ), device=x.device)
#     print(x.device, t.device)
#     # t_real = torch.randint(T, size=(x.shape[0],), device=x.device, dtype=torch.float)
#     # t_imag = torch.randint(T, size=(x.shape[0],), device=x.device, dtype=torch.float)
#     # t = t_real + 1j * t_imag
#     model = UNet(in_channel=2, out_channel=1, T=T, inner_channel=64, 
#                  channel_mults=[1,2,3,4], attn_res=[2], num_res_blocks=2, dropout=0.2)
#     out = model(x, t)