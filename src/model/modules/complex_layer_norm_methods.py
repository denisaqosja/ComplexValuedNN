"""
In this file, we implement various normalization layers for complex-valued tensors, including:  
- ComplexGroupNorm: Group Normalization using true complex whitening via 2x2 covariance matrix.
- ComplexBatchNorm: Batch Normalization using full 2x2 covariance matrix for running statistics and affine transformation.
- ComplexBatchNorm2D_bmm: Batch Normalization using batch matrix multiplication for efficient covariance computation 
and whitening (slower in computational speed).
"""

import torch
import torch.nn as nn
from torch.nn import Parameter


class ComplexGroupNorm(nn.Module):
    """
    Group Normalization for complex-valued tensors using true complex
    normalization via 2x2 covariance matrix whitening.

    Each group is whitened using the analytic inverse square root of its
    2x2 covariance matrix [[Vrr, Vri], [Vri, Vii]], preserving the
    relationship between real and imaginary components.

    Args:
        num_groups  (int)  : Number of groups to divide channels into.
        num_channels (int) : Number of channels in the input.
        eps         (float): Small value for numerical stability. Default: 1e-5.
        affine      (bool) : If True, learns affine parameters. Default: True.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int=32,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()

        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight_real = Parameter(torch.ones(num_channels))
            self.weight_imag = Parameter(torch.ones(num_channels))
            self.bias_real   = Parameter(torch.zeros(num_channels))
            self.bias_imag   = Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape (B, C, *spatial) with dtype torch.cfloat
               or torch.cdouble.

        Returns:
            Normalized complex tensor of the same shape and dtype.
        """
        if not x.is_complex():
            raise TypeError(f"Expected a complex tensor, got {x.dtype}")

        real, imag = x.real, x.imag                        # (B, C, *spatial)
        B, C = real.shape[:2]
        spatial = real.shape[2:]
        G  = self.num_groups
        CG = C // G

        # ── reshape into groups: (B, G, C//G, *spatial) ──────────────────────
        real_g = real.view(B, G, CG, *spatial)
        imag_g = imag.view(B, G, CG, *spatial)

        # ── per-group mean (reduce over C//G and all spatial dims) ────────────
        reduce_dims = list(range(2, real_g.ndim))           # [2, 3, ...]
        mu_r = real_g.mean(dim=reduce_dims, keepdim=True)
        mu_i = imag_g.mean(dim=reduce_dims, keepdim=True)
        
        r = real_g - mu_r
        i = imag_g - mu_i

        # ── 2x2 covariance matrix entries ─────────────────────────────────────
        Vrr = (r * r).mean(dim=reduce_dims, keepdim=True) + self.eps
        Vii = (i * i).mean(dim=reduce_dims, keepdim=True) + self.eps
        Vri = (r * i).mean(dim=reduce_dims, keepdim=True)

        # ── analytic inverse square root of the 2x2 symmetric PD matrix ──────
        # Given M = [[Vrr, Vri], [Vri, Vii]], we want M^{-1/2}.
        # Let  τ = sqrt(det(M)) = sqrt(Vrr*Vii - Vri²)
        #      s = sqrt(τ * (Vrr + Vii + 2τ))   [= sqrt(tr(M + τI) * τ)]
        # Then M^{-1/2} = (1/s) * [[Vii+τ, -Vri], [-Vri, Vrr+τ]]
        tau = (Vrr * Vii - Vri ** 2).clamp(min=0).sqrt()
        s   = (tau * (Vrr + Vii + 2 * tau)).clamp(min=self.eps).sqrt()

        real_norm = (r * (Vii + tau) - i * Vri) / s
        imag_norm = (i * (Vrr + tau) - r * Vri) / s

        # ── reshape back to (B, C, *spatial) ─────────────────────────────────
        real_norm = real_norm.view(B, C, *spatial)
        imag_norm = imag_norm.view(B, C, *spatial)

        # ── learnable affine transform ────────────────────────────────────────
        if self.affine:
            shape = (1, C) + (1,) * len(spatial)
            real_norm = real_norm * self.weight_real.view(shape) + self.bias_real.view(shape)
            imag_norm = imag_norm * self.weight_imag.view(shape) + self.bias_imag.view(shape)

        return torch.complex(real_norm, imag_norm)

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, num_channels={self.num_channels}, "
            f"eps={self.eps}, affine={self.affine}"
        )
    

class ComplexGroupNorm(nn.Module):
    """
    Complex Group Normalization for 2D complex-valued tensors.
    
    This implementation normalizes across groups of channels
    It uses:
    - Separate weight and bias parameters for real and imaginary parts (gamma_real, gamma_imag, beta_real, beta_imag)
    - Running mean and variance for real and imaginary parts (running_mean_real, running_mean_imag, running_var_real, running_var_imag)
    
    Args:
        num_features: Number of channels (C)
        num_groups: Number of groups to divide channels into (default: 32)
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics (default: 0.1)
        affine: If True, applies learnable affine transformation (default: True)
        track_running_stats: If True, tracks running mean/variance (default: True)
    """
    def __init__(self, num_features, num_groups=32, eps=1e-5, momentum=0.1, affine=True):
        super(ComplexGroupNorm, self).__init__()
        
        self.num_channels = num_features 
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({self.num_channels}) must be divisible by num_groups ({num_groups})"
            )

        if affine:
            # Scale and shift parameters for complex numbers
            # gamma is a 2x2 matrix per channel
            self.gamma = nn.Parameter(torch.eye(2).repeat(num_features, 1, 1))  # [C, 2, 2]
            self.beta = nn.Parameter(torch.zeros(num_features, 2))              # [C, 2]
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset all parameters."""
        if self.affine:
            # Initialize weight matrix as identity
            # This means: no scaling, no rotation initially
            self.gamma.data.zero_()
            self.gamma.data[:, 0, 0] = 1.0  # w_rr = 1
            self.gamma.data[:, 1, 1] = 1.0  # w_ii = 1
            # Off-diagonal stays 0 (w_ri = w_ir = 0)
            
            # Initialize bias as zero
            self.beta.data.zero_()

    def forward(self, x):
        N, C = x.shape[:2]
        G = self.num_groups

        # reshape
        x = x.view(N, G, C // G, *x.shape[2:])

        # Split into real and imaginary parts
        x_real = x.real
        x_imag = x.imag

        reduce_dims = list(range(2, x_real.ndim))

        # Means
        mu_r = x_real.mean(dim=reduce_dims, keepdim=True)
        mu_i = x_imag.mean(dim=reduce_dims, keepdim=True)


        
        # calculate mean
        mean_r = x_real.mean(dim=(2, *range(3, x.dim())), keepdim=True)
        mean_i = x_imag.mean(dim=(2, *range(3, x.dim())), keepdim=True)

        xr = x_real - mean_r
        xi = x_imag - mean_i
        # calculate covariance matrix
        var_rr = (xr * xr).mean(dim=(2, *range(3, x.dim())), keepdim=True)
        var_ii = (xi * xi).mean(dim=(2, *range(3, x.dim())), keepdim=True)
        cov_ri = (xr * xi).mean(dim=(2, *range(3, x.dim())), keepdim=True)

        det = var_rr * var_ii - cov_ri**2 + self.eps
        
        inv_sqrt_rr =  var_ii / torch.sqrt(det)
        inv_sqrt_ii =  var_rr / torch.sqrt(det)
        inv_sqrt_ri = -cov_ri / torch.sqrt(det)

        xr_norm = inv_sqrt_rr * xr + inv_sqrt_ri * xi
        xi_norm = inv_sqrt_ri * xr + inv_sqrt_ii * xi

        x_norm = torch.complex(xr_norm, xi_norm)

        return x_norm.view(N, C, *x.shape[3:])



class ComplexBatchNorm(nn.Module):
    """
    Complex Batch Normalization for 2D complex-valued tensors.
    
    This version uses:
    - Full (n_features, 2, 2) covariance matrix for running statistics
    - Full (n_features, 2, 2) weight matrix for affine transformation
    - (n_features, 2) bias vector
    
    Args:
        num_features: Number of channels (C)
        eps: Small constant for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics (default: 0.1)
        affine: If True, applies learnable affine transformation (default: True)
        track_running_stats: If True, tracks running mean/covariance (default: True)
    
    Shape:
        Input: (N, C, H, W) complex tensor
        Output: (N, C, H, W) complex tensor
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True):
        super(ComplexBatchNorm, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Learnable affine parameters as full matrices
        if self.affine:
            # Gamma - Weight matrix: (n_features, 2, 2)
            # For each feature i:
            #   weight[i] = [[w_rr, w_ri],
            #                [w_ir, w_ii]]
            # where w_ir = w_ri for symmetry in this implementation
            self.gamma = Parameter(torch.zeros(num_features, 2, 2))
            
            # Beta - Bias vector: (n_features, 2) where [:, 0] is real, [:, 1] is imag
            self.beta = Parameter(torch.zeros(num_features, 2))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        
        # Running statistics with full covariance matrix
        if self.track_running_stats:
            # Mean vector: (n_features, 2) where [:, 0] is real, [:, 1] is imag
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            
            # Covariance matrix: (n_features, 2, 2)
            # For each feature i:
            #   running_cov[i] = [[Var(real), Cov(real,imag)],
            #                     [Cov(real,imag), Var(imag)]]
            self.register_buffer('running_cov', torch.zeros(num_features, 2, 2))
            
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_cov', None)
            self.register_buffer('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_running_stats(self):
        """Reset running statistics to initial values."""
        if self.track_running_stats:
            # Mean starts at zero
            self.running_mean.zero_()
            
            # Covariance matrix initialized as identity matrix
            self.running_cov.zero_()
            self.running_cov[:, 0, 0] = 1.0  # Var(real) = 1
            self.running_cov[:, 1, 1] = 1.0  # Var(imag) = 1
            # Off-diagonal stays 0 (no correlation)
            
            self.num_batches_tracked.zero_()
    
    def reset_parameters(self):
        """Reset all parameters."""
        self.reset_running_stats()
        if self.affine:
            # Initialize weight matrix as identity
            # This means: no scaling, no rotation initially
            self.gamma.data.zero_()
            self.gamma.data[:, 0, 0] = 1.0  # w_rr = 1
            self.gamma.data[:, 1, 1] = 1.0  # w_ii = 1
            # Off-diagonal stays 0 (w_ri = w_ir = 0)
            
            # Initialize bias as zero
            self.beta.data.zero_()
    
    def forward(self, input):
        # Check input
        assert input.dim() == 4, f'Expected 4D input (got {input.dim()}D)'
        assert input.is_complex(), 'Expected complex-valued input'
        
        # Update batch counter
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        
        # Split into real and imaginary parts
        input_real = input.real
        input_imag = input.imag
        
        # Compute or use statistics
        if self.training or not self.track_running_stats:
            # Compute batch statistics
            mean_real = input_real.mean(dim=[0, 2, 3])
            mean_imag = input_imag.mean(dim=[0, 2, 3])
            
            # Center the data
            centered_real = input_real - mean_real.view(1, -1, 1, 1)
            centered_imag = input_imag - mean_imag.view(1, -1, 1, 1)
            
            # Compute covariance matrix components
            n_elements = input_real.numel() / self.num_features
            cov_rr = (centered_real ** 2).sum(dim=[0, 2, 3]) / n_elements
            cov_ii = (centered_imag ** 2).sum(dim=[0, 2, 3]) / n_elements
            cov_ri = (centered_real * centered_imag).sum(dim=[0, 2, 3]) / n_elements
            
            # Update running statistics
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    # Update mean
                    new_mean = torch.stack([mean_real, mean_imag], dim=1)  # (n_features, 2)
                    self.running_mean = (
                        exponential_average_factor * new_mean +
                        (1 - exponential_average_factor) * self.running_mean
                    )
                    
                    # Update covariance matrix
                    batch_cov = torch.zeros(self.num_features, 2, 2, 
                                           device=input.device, dtype=input_real.dtype)
                    batch_cov[:, 0, 0] = cov_rr
                    batch_cov[:, 0, 1] = cov_ri
                    batch_cov[:, 1, 0] = cov_ri  # Symmetric
                    batch_cov[:, 1, 1] = cov_ii
                    
                    self.running_cov = (
                        exponential_average_factor * batch_cov +
                        (1 - exponential_average_factor) * self.running_cov
                    )
        else:
            # Use running statistics
            mean_real = self.running_mean[:, 0]
            mean_imag = self.running_mean[:, 1]
            cov_rr = self.running_cov[:, 0, 0]
            cov_ri = self.running_cov[:, 0, 1]
            cov_ii = self.running_cov[:, 1, 1]
            
            centered_real = input_real - mean_real.view(1, -1, 1, 1)
            centered_imag = input_imag - mean_imag.view(1, -1, 1, 1)
        
        # Add epsilon for numerical stability
        cov_rr = cov_rr + self.eps
        cov_ii = cov_ii + self.eps
        
        # Compute inverse square root of covariance matrix
        det = cov_rr * cov_ii - cov_ri ** 2
        det = torch.clamp(det, min=self.eps)
        
        s = torch.sqrt(det)
        t = torch.sqrt(cov_rr + 2 * s + cov_ii)
        
        rst = 1.0 / (s * t)
        inv_sqrt_rr = (cov_ii + s) * rst
        inv_sqrt_ri = -cov_ri * rst
        inv_sqrt_ii = (cov_rr + s) * rst
        
        # Apply whitening transformation
        normalized_real = (inv_sqrt_rr.view(1, -1, 1, 1) * centered_real +
                          inv_sqrt_ri.view(1, -1, 1, 1) * centered_imag)
        normalized_imag = (inv_sqrt_ri.view(1, -1, 1, 1) * centered_real +
                          inv_sqrt_ii.view(1, -1, 1, 1) * centered_imag)
        
        # Apply affine transformation using weight matrix
        if self.affine:
            # Extract weight matrix components
            w_rr = self.gamma[:, 0, 0]  # (n_features,)
            w_ri = self.gamma[:, 0, 1]  # (n_features,)
            w_ir = self.gamma[:, 1, 0]  # (n_features,)
            w_ii = self.gamma[:, 1, 1]  # (n_features,)
            
            # Extract bias components
            b_real = self.beta[:, 0]  # (n_features,)
            b_imag = self.beta[:, 1]  # (n_features,)
            
            # Apply transformation: [out_real, out_imag]^T = W * [norm_real, norm_imag]^T + b
            output_real = (w_rr.view(1, -1, 1, 1) * normalized_real +
                          w_ri.view(1, -1, 1, 1) * normalized_imag +
                          b_real.view(1, -1, 1, 1))
            output_imag = (w_ir.view(1, -1, 1, 1) * normalized_real +
                          w_ii.view(1, -1, 1, 1) * normalized_imag +
                          b_imag.view(1, -1, 1, 1))
        else:
            output_real = normalized_real
            output_imag = normalized_imag
        
        return torch.complex(output_real, output_imag)
    
    def extra_repr(self):
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}')



class ComplexBatchNorm2D_bmm(nn.Module):
    """
    This implementation of Complex Batch Normalization uses batch matrix multiplication for efficient computation of covariance and whitening.
        - Running covariance is stored as a full 2x2 matrix per feature channel.
        - Affine transformation is applied using a full 2x2 weight matrix per feature channel, allowing for scaling and rotation in the complex plane.
    Because of eigenvalue decomposition + bmm for whitening, this implementation is more computationally intensive than the other ComplexBatchNorm2d, 
    (but it is more mathematically rigorous and can capture correlations between real and imaginary parts more effectively).
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # keep afine always set to True; makes the shifting and scaling after whitening/normalization
        self.affine = affine
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        if affine:
            # Scale and shift parameters for complex numbers
            # gamma is a 2x2 matrix per channel
            self.gamma = nn.Parameter(torch.eye(2).repeat(num_features, 1, 1))  # [C, 2, 2]
            self.beta = nn.Parameter(torch.zeros(num_features, 2))              # [C, 2]
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        if self.track_running_stats:
            # Running statistics  - mean and covariance matrix
            self.register_buffer("running_mean", torch.zeros(num_features, 1, 2))
            # 2x2 covariance matrix per channel
            self.register_buffer("running_cov", torch.eye(2).repeat(num_features, 1, 1))

            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_cov', None)
            self.register_buffer('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_running_stats(self):
        """Reset running statistics to initial values."""
        if self.track_running_stats:
            self.running_mean.zero_()
            # initialize with unit variance I
            self.running_cov.zero_()
            # Set diagonal to 1 (unit variance for both real and imag)
            self.running_cov[:, 0, 0] = 1.0  # Var(real) = 1
            self.running_cov[:, 1, 1] = 1.0  # Var(imag) = 1
            # Off-diagonal stays 0 (no correlation initially)

            self.num_batches_tracked.zero_()
    
    def reset_parameters(self):
        """Reset learnable parameters and running statistics."""
        self.reset_running_stats()
        if self.affine:
            # Initialize weight matrix as IDENTITY
            self.gamma.data.zero_()
            self.gamma.data[:, 0, 0] = 1.0  # w_rr = 1
            self.gamma.data[:, 1, 1] = 1.0  # w_ii = 1
            # Off-diagonal stays 0 (w_ri = w_ir = 0) no correlation 
        
            # Initialize bias as ZEROS
            self.beta.data.zero_()

    def _check_input_dim(self, x: torch.Tensor):
        """Verify input has correct dimensions."""
        if x.dim() != 4:
            raise ValueError(f'Expected 4D input (got {x.dim()}D input)')
        if not x.is_complex():
            raise ValueError('Expected complex-valued input tensor')

    def forward(self, x):
        """
        Apply complex batch normalization.
        Args:
            input: Complex tensor of shape (N, C, H, W)
        Returns:
            Normalized complex tensor of same shape
        """
        self._check_input_dim(x)

        # Determine if we should use running stats or compute batch stats
        exponential_average_factor = 0.0
        
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    exponential_average_factor = self.momentum
        
        # stack along the last axis
        x = torch.stack([x.real, x.imag], dim=-1)   # B, C, H, W, 2
        B, C, Hg, Wd, _ = x.shape

        x_perm = x.permute(1, 0, 2, 3, 4)       # [C, B, H, W, 2]
        x_perm = x_perm.reshape(C, -1, 2)       # [C, N, 2]  where N = B*H*W
        
        if self.training or not self.track_running_stats:
            # ---- Compute batch statistics ----
            # Compute mean
            mean = x_perm.mean(dim=1, keepdim=True)  # [C,1,2]
            x_centered = x_perm - mean               # [C,N,2]
            # breakpoint()
            
            # Compute covariance (vectorized)
            # x_centered: [C, N, 2]
            # Compute covariance matrix for all channels at once
            N = x_centered.shape[1]
            # Use batch matrix multiplication: [C, 2, N] @ [C, N, 2] = [C, 2, 2]
            cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / N
            # Add epsilon * I to all covariance matrices
            cov = cov + self.eps * torch.eye(2, dtype=x_perm.dtype, device=x_perm.device).unsqueeze(0)

            # ---- Update running statistics ----
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = (
                            exponential_average_factor * mean + 
                            (1 - exponential_average_factor) * self.running_mean
                    )
                    self.running_cov = (
                        exponential_average_factor * cov + 
                        (1 - exponential_average_factor) * self.running_cov
                    )
        else:
            # ---- Use stored running statistics ----
            mean = self.running_mean
            cov = self.running_cov
            x_centered = x_perm - mean     

        # Whitening using eigendecomposition (vectorized)
        # Covariance matrix: [[Vrr, Vri], [Vri, Vii]]
        # We need to compute the inverse square root of this matrix
        eigvals, eigvecs = torch.linalg.eigh(cov)  # eigvals: [C, 2], eigvecs: [C, 2, 2]
        # Compute D_inv_sqrt: [C, 2, 2] diagonal matrices with (eigvals + eps)^(-0.5)
        D_inv_sqrt = torch.diag_embed((eigvals + self.eps).pow(-0.5))  # [C, 2, 2]
        # W = eigvecs @ D_inv_sqrt @ eigvecs.T for each channel
        # Using batch matrix multiplication: [C, 2, 2] @ [C, 2, 2] @ [C, 2, 2]
        W = torch.bmm(torch.bmm(eigvecs, D_inv_sqrt), eigvecs.transpose(1, 2))  # [C, 2, 2]
        
        # Apply whitening: x_centered [C, N, 2] @ W.T [C, 2, 2] -> [C, N, 2]
        x_norm = torch.bmm(x_centered, W.transpose(1, 2))  # [C, N, 2]
        
        # Affine transform (vectorized)
        if self.affine:
            # x_norm @ gamma.T + beta for each channel
            # x_norm: [C, N, 2], gamma: [C, 2, 2], beta: [C, 2]
            x_norm = torch.bmm(x_norm, self.gamma.transpose(1, 2)) + self.beta.unsqueeze(1)  # [C, N, 2]
            
        # Reshape back
        x_norm = x_norm.reshape(C, B, Hg, Wd, 2)
        x_norm = x_norm.permute(1, 0, 2, 3, 4)    # [B,C,H,W,2]

        x_complex = torch.complex(x_norm[..., 0], x_norm[..., 1])
        return x_complex

#

# # Demonstration
# if __name__ == "__main__":
#     print("ComplexBatchNorm2d with Full Matrix Representation\n" + "="*70)
    
#     # Create layer
#     bn = ComplexBatchNorm(num_features=4)
#     print(f"Layer: {bn}\n")
    
#     # Show parameter shapes
#     print("Parameter Shapes:")
#     print(f"  gamma: {bn.gamma.shape}")
#     print(f"  bias: {bn.beta.shape}")
#     print(f"  running_mean: {bn.running_mean.shape}")
#     print(f"  running_cov: {bn.running_cov.shape}")
    
#     # Show initial weight matrix
#     print("\n" + "="*70)
#     print("Initial Gamma Matrix (Identity)")
#     print("="*70)
#     for i in range(min(2, bn.num_features)):
#         print(f"\nFeature {i}:")
#         print(f"  gamma[{i}] = [[{bn.gamma[i, 0, 0]:.4f}, {bn.gamma[i, 0, 1]:.4f}],")
#         print(f"                  [{bn.gamma[i, 1, 0]:.4f}, {bn.gamma[i, 1, 1]:.4f}]]")
#         print(f"  beta[{i}] = [{bn.beta[i, 0]:.4f}, {bn.beta[i, 1]:.4f}]")
    
#     print("\nInterpretation:")
#     print("  - Identity gamma matrix means no transformation initially")
#     print("  - During training, network learns optimal transformation")
    
#     # Show initial covariance
#     print("\n" + "="*70)
#     print("Initial Covariance Matrix (Identity)")
#     print("="*70)
#     for i in range(min(2, bn.num_features)):
#         print(f"\nFeature {i}:")
#         print(f"  running_cov[{i}] = [[{bn.running_cov[i, 0, 0]:.4f}, {bn.running_cov[i, 0, 1]:.4f}],")
#         print(f"                       [{bn.running_cov[i, 1, 0]:.4f}, {bn.running_cov[i, 1, 1]:.4f}]]")
    
#     # Test forward pass
#     print("\n" + "="*70)
#     print("Forward Pass Test")
#     print("="*70)
    
#     bn.train()
#     x = torch.randn(8, 4, 16, 16, dtype=torch.complex64)
#     y = bn(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
    
#     # Count parameters
#     total_params = sum(p.numel() for p in bn.parameters() if p.requires_grad)
#     print(f"\nTotal learnable parameters: {total_params}")
#     print(f"  Gamma matrix: {bn.gamma.numel()} (4 per feature)")
#     print(f"  Beta vector: {bn.beta.numel()} (2 per feature)")
#     print(f"  Total per feature: 6")
    
#     # After training, weight might look different
#     print("\n" + "="*70)
#     print("After Training (Simulated)")
#     print("="*70)
    
#     # Simulate some gradient updates
#     optimizer = torch.optim.SGD(bn.parameters(), lr=0.1)
#     for _ in range(5):
#         x_train = torch.randn(16, 4, 16, 16, dtype=torch.complex64)
#         y_train = bn(x_train)
#         loss = y_train.abs().mean()
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
    
#     print("gamma matrix after training (feature 0):")
#     print(f"  gamma[0] = [[{bn.gamma[0, 0, 0]:.4f}, {bn.gamma[0, 0, 1]:.4f}],")
#     print(f"                [{bn.gamma[0, 1, 0]:.4f}, {bn.gamma[0, 1, 1]:.4f}]]")
#     print(f"  beta[0] = [{bn.beta[0, 0]:.4f}, {bn.beta[0, 1]:.4f}]")
    
#     print("\nNote: Gamma/Weight has learned to scale/rotate the normalized features")
    
#     print("\n✓ Test completed!")
