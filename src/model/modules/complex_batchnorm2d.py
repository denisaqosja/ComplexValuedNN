import torch
import torch.nn as nn
from torch.nn import Parameter


class ComplexBatchNorm2d(nn.Module):
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
        super(ComplexBatchNorm2d, self).__init__()
        
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


# Demonstration
if __name__ == "__main__":
    print("ComplexBatchNorm2d with Full Matrix Representation\n" + "="*70)
    
    # Create layer
    bn = ComplexBatchNorm2d(num_features=4)
    print(f"Layer: {bn}\n")
    
    # Show parameter shapes
    print("Parameter Shapes:")
    print(f"  gamma: {bn.gamma.shape}")
    print(f"  bias: {bn.beta.shape}")
    print(f"  running_mean: {bn.running_mean.shape}")
    print(f"  running_cov: {bn.running_cov.shape}")
    
    # Show initial weight matrix
    print("\n" + "="*70)
    print("Initial Gamma Matrix (Identity)")
    print("="*70)
    for i in range(min(2, bn.num_features)):
        print(f"\nFeature {i}:")
        print(f"  gamma[{i}] = [[{bn.gamma[i, 0, 0]:.4f}, {bn.gamma[i, 0, 1]:.4f}],")
        print(f"                  [{bn.gamma[i, 1, 0]:.4f}, {bn.gamma[i, 1, 1]:.4f}]]")
        print(f"  beta[{i}] = [{bn.beta[i, 0]:.4f}, {bn.beta[i, 1]:.4f}]")
    
    print("\nInterpretation:")
    print("  - Identity gamma matrix means no transformation initially")
    print("  - During training, network learns optimal transformation")
    
    # Show initial covariance
    print("\n" + "="*70)
    print("Initial Covariance Matrix (Identity)")
    print("="*70)
    for i in range(min(2, bn.num_features)):
        print(f"\nFeature {i}:")
        print(f"  running_cov[{i}] = [[{bn.running_cov[i, 0, 0]:.4f}, {bn.running_cov[i, 0, 1]:.4f}],")
        print(f"                       [{bn.running_cov[i, 1, 0]:.4f}, {bn.running_cov[i, 1, 1]:.4f}]]")
    
    # Test forward pass
    print("\n" + "="*70)
    print("Forward Pass Test")
    print("="*70)
    
    bn.train()
    x = torch.randn(8, 4, 16, 16, dtype=torch.complex64)
    y = bn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in bn.parameters() if p.requires_grad)
    print(f"\nTotal learnable parameters: {total_params}")
    print(f"  Gamma matrix: {bn.gamma.numel()} (4 per feature)")
    print(f"  Beta vector: {bn.beta.numel()} (2 per feature)")
    print(f"  Total per feature: 6")
    
    # After training, weight might look different
    print("\n" + "="*70)
    print("After Training (Simulated)")
    print("="*70)
    
    # Simulate some gradient updates
    optimizer = torch.optim.SGD(bn.parameters(), lr=0.1)
    for _ in range(5):
        x_train = torch.randn(16, 4, 16, 16, dtype=torch.complex64)
        y_train = bn(x_train)
        loss = y_train.abs().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print("gamma matrix after training (feature 0):")
    print(f"  gamma[0] = [[{bn.gamma[0, 0, 0]:.4f}, {bn.gamma[0, 0, 1]:.4f}],")
    print(f"                [{bn.gamma[0, 1, 0]:.4f}, {bn.gamma[0, 1, 1]:.4f}]]")
    print(f"  beta[0] = [{bn.beta[0, 0]:.4f}, {bn.beta[0, 1]:.4f}]")
    
    print("\nNote: Gamma/Weight has learned to scale/rotate the normalized features")
    
    print("\n✓ Test completed!")
