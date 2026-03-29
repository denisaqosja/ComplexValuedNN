
import torch 
import torch.nn as nn

"""
Code proved to show same results as the pytorch implementation torch.nn.BatchNorm2D
"""


class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Learnable parameters (gamma and beta)
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # Running statistics (buffers, not trainable)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Expected 4D input (N, C, H, W)")

        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var
                )
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean
            var = self.running_var

        # Reshape for broadcasting
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation
        if self.affine:
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x_hat = x_hat * weight + bias

        return x_hat