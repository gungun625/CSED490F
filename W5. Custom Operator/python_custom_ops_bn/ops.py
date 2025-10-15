import torch
from torch import Tensor
from typing import Tuple

# Step 1: Define custom operators using torch.library API
@torch.library.custom_op("my_ops::batchnorm_forward", mutates_args=("running_mean", "running_var"))
def batchnorm_forward(
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    beta: Tensor,            # [C]
    running_mean: Tensor,    # [C]
    running_var: Tensor,     # [C]
    training: bool,
    momentum: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """forward pass of BatchNorm for 4D input [N, C, H, W]."""

    # Implement Here
    reduce_dims = (0, 2, 3)
    def _chan_view(t: Tensor) -> Tensor:
        return t.view(1, -1, 1, 1)

    if training:
        batch_mean = input.mean(dim=reduce_dims)                      # [C]
        batch_var  = input.var(dim=reduce_dims, unbiased=False)       # [C]
        invstd     = torch.rsqrt(batch_var + eps)                     # [C]

        # running = (1 - m) * running + m * batch_stat
        running_mean.mul_(1.0 - momentum).add_(momentum * batch_mean)
        running_var.mul_(1.0 - momentum).add_(momentum * batch_var)

        mean_for_norm   = batch_mean
        invstd_for_norm = invstd

        save_mean   = batch_mean.detach()
        save_invstd = invstd.detach()
    else:
        mean_for_norm   = running_mean.detach().clone()
        invstd_for_norm = torch.rsqrt(running_var + eps)

        save_mean   = running_mean.detach().clone()
        save_invstd = invstd_for_norm.detach().clone()

    x_hat  = (input - _chan_view(mean_for_norm)) * _chan_view(invstd_for_norm)  # [N,C,H,W]
    output = _chan_view(gamma) * x_hat + _chan_view(beta)                       # [N,C,H,W]
    
    return output, save_mean, save_invstd


@torch.library.custom_op("my_ops::batchnorm_backward", mutates_args=())
def batchnorm_backward(
    grad_output: Tensor,     # [N, C, H, W]
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    save_mean: Tensor,       # [C]
    save_invstd: Tensor      # [C]
) -> Tuple[Tensor, Tensor, Tensor]:
    """backward pass of BatchNorm for 4D input."""

    # Implement Here
    C = input.shape[1]
    N_H_W = torch.tensor(input.shape[0] * input.shape[2] * input.shape[3], 
                         dtype=input.dtype, device=input.device)
    
    x_hat = (input - save_mean.view(1, -1, 1, 1)) * save_invstd.view(1, -1, 1, 1)
    
    # Gradient w.r.t. beta (shift)
    grad_beta = grad_output.sum(dim=(0, 2, 3))
    
    # Gradient w.r.t. gamma (scale)
    grad_gamma = (grad_output * x_hat).sum(dim=(0, 2, 3))
    
    # Gradient w.r.t. input (x)
    dL_dx_hat = grad_output * gamma.view(1, -1, 1, 1)
    
    mean_dL_dx_hat = dL_dx_hat.sum(dim=(0, 2, 3)).view(1, -1, 1, 1)
    mean_dL_dx_hat_x_hat = (dL_dx_hat * x_hat).sum(dim=(0, 2, 3)).view(1, -1, 1, 1)
    
    grad_input = (save_invstd.view(1, -1, 1, 1) / N_H_W) * (
        N_H_W * dL_dx_hat 
        - mean_dL_dx_hat 
        - x_hat * mean_dL_dx_hat_x_hat
    )

    return grad_input, grad_gamma, grad_beta


# Step 2: Connect forward and backward with autograd
# This connects our custom forward/backward operators to PyTorch's 
# autograd system, allowing gradients to flow during backpropagation
class BatchNormCustom(torch.autograd.Function):
    """
    Custom Batch Normalization for 4D inputs [N, C, H, W].
    
    Bridges custom operators with PyTorch's autograd engine.
    - forward(): calls custom forward operator and saves context
    - backward(): calls custom backward operator using saved context

    Usage:
        output = BatchNormCustom.apply(input, gamma, beta, running_mean, running_var, training, momentum, eps)
    """
    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var, training, momentum, eps):
        output, save_mean, save_invstd = torch.ops.my_ops.batchnorm_forward(
            input, gamma, beta, running_mean, running_var, training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.my_ops.batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd
        )
        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None