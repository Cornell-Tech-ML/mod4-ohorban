from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling.

    Args:
    ----
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): Pooling kernel size as (height, width).

    Returns:
    -------
        Tuple[Tensor, int, int]: Reshaped tensor of size
        (batch, channel, new_height, new_width, kernel_height * kernel_width),
        and the new height and width.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    reshaped = input.contiguous().view(batch, channel, height, new_width, kw)
    reshaped = reshaped.permute(0, 1, 3, 2, 4)
    reshaped = reshaped.contiguous().view(
        batch, channel, new_height, new_width, kh * kw
    )

    return reshaped, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to a 2D input tensor.

    Args:
    ----
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): Pooling kernel size as (height, width).

    Returns:
    -------
        Tensor: Output tensor of shape (batch, channel, new_height, new_width).

    """
    batch, channel, _, _ = input.shape
    reshaped, new_height, new_width = tile(input, kernel)
    pooled = reshaped.mean(dim=4)
    return pooled.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input (Tensor): Tensor to compute argmax on.
        dim (int): Dimension to reduce.

    Returns:
    -------
        Tensor: One-hot encoded tensor with the argmax.

    """
    reduced_max = max_reduce(input, dim)
    return reduced_max == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for the Max function.

        Args:
        ----
            ctx (Context): Context to save information for backward computation.
            input (Tensor): Input tensor to apply the max function on.
            dim (Tensor): Dimension to reduce.

        Returns:
        -------
            Tensor: Tensor after applying the max function along the specified dimension.

        """
        ctx.save_for_backward(input, dim)
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the Max function.

        Args:
        ----
            ctx (Context): Context with saved information from the forward pass.
            grad_output (Tensor): Gradient of the loss with respect to the output.

        Returns:
        -------
            Tuple[Tensor, float]: Gradients with respect to the input tensor and dimension.

        """
        input, dim = ctx.saved_values
        return (argmax(input, int(dim.item())) * grad_output, dim)


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
    ----
        input (Tensor): Tensor to apply the max function on.
        dim (int): Dimension to reduce.

    Returns:
    -------
        Tensor: Tensor after applying the max function.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a specified dimension.

    Args:
    ----
        input (Tensor): Tensor to compute the softmax on.
        dim (int): Dimension to reduce.

    Returns:
    -------
        Tensor: Tensor after applying softmax.

    """
    exp_tensor = input.exp()
    return exp_tensor / exp_tensor.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along a specified dimension.

    Args:
    ----
        input (Tensor): Tensor to compute the logsoftmax on.
        dim (int): Dimension to reduce.

    Returns:
    -------
        Tensor: Tensor after applying logsoftmax.

    """
    max_vals = max(input, dim)
    return input - ((input - max_vals).exp().sum(dim=dim).log() + max_vals)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to a 2D input tensor.

    Args:
    ----
        input (Tensor): Input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): Pooling kernel size as (height, width).

    Returns:
    -------
        Tensor: Output tensor of shape (batch, channel, new_height, new_width).

    """
    batch, channel, _, _ = input.shape
    reshaped, new_height, new_width = tile(input, kernel)
    pooled = max(reshaped, 4)
    return pooled.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor.

    Args:
    ----
        input (Tensor): Tensor to apply dropout on.
        rate (float): Dropout rate, the probability of setting a unit to zero.
        ignore (bool): If True, dropout is not applied.

    Returns:
    -------
        Tensor: Tensor after applying dropout.

    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
