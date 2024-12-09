"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Tuple

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend


if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Any) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Any) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False

        for v in vals:
            if isinstance(v, minitorch.Tensor):
                if v.requires_grad():
                    need_grad = True
                raw_vals.append(v.detach())
            else:
                # For non-Tensor arguments, append them directly
                raw_vals.append(v)

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes forward pass for negation."""
        ctx.save_for_backward(a)
        return a.f.neg_map(
            a, zeros(a.shape, backend=a.backend)
        )  # I couldn't figure out how to not pass zero tensors for map functions

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes backward pass for negation."""
        return -grad_output


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes forward pass for inversion."""
        ctx.save_for_backward(a)
        return a.f.inv_map(a, zeros(a.shape, backend=a.backend))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes backward pass for inversion."""
        (a,) = ctx.saved_values
        return grad_output.f.inv_back_zip(a, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes forward pass for addition."""
        return a.f.add_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes backward pass for addition."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes forward pass for multiplication."""
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes backward pass for multiplication."""
        a, b = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, b), grad_output.f.mul_zip(
            grad_output, a
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes forward pass for sigmoid."""
        out = a.f.sigmoid_map(a, zeros(a.shape, backend=a.backend))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes backward pass for sigmoid."""
        out = ctx.saved_values[0]
        return grad_output * out * (1 - out)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes forward pass for ReLU."""
        ctx.save_for_backward(a)
        return a.f.relu_map(a, zeros(a.shape, backend=a.backend))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes backward pass for ReLU."""
        a = ctx.saved_values[0]
        return grad_output * (a > 0)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes forward pass for natural logarithm."""
        ctx.save_for_backward(a)
        result = a.f.log_map(a, zeros(a.shape, backend=a.backend))
        # print(f"Forward pass: input {a}, output {result}")
        return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes backward pass for natural logarithm."""
        a = ctx.saved_values[0]
        result = grad_output.f.mul_zip(grad_output, a.f.inv_map(a))
        # print(f"Backward pass: input {a}, output {result}")
        return result


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Computes forward pass for exponential function."""
        out = a.f.exp_map(a, zeros(a.shape, backend=a.backend))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Computes backward pass for exponential function."""
        a = ctx.saved_values[0]
        return grad_output.f.mul_zip(a, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes forward pass for sum."""
        dim_value = int(dim.item())
        ctx.save_for_backward(a, dim)
        return a.f.add_reduce(a, dim_value)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Computes backward pass for sum."""
        a, dim = ctx.saved_values
        dim_value = int(dim.item())
        shape_list = list(a.shape)
        shape_list[dim_value] = 1

        grad_output_reshaped = grad_output.view(*shape_list)

        ones_tensor = zeros(a.shape, backend=a.backend) + 1.0
        grad_input = grad_output_reshaped * ones_tensor
        zero_grad = zeros(dim.shape, backend=dim.backend)
        return grad_input, zero_grad


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes forward pass for 'less than' comparison."""
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        """Computes backward pass for 'less than' comparison."""
        return None, None


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes forward pass for element-wise equality."""
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        """Computes backward pass for element-wise equality."""
        return None, None


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Computes forward pass for is_close."""
        return a.f.is_close_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        """Computes backward pass for is_close."""
        return None, None


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Computes forward pass for permute."""
        ctx.save_for_backward(order)
        order_list = [int(order[i]) for i in range(order.shape[0])]
        assert len(order_list) == len(
            a.shape
        ), "Order must match the number of dimensions."
        assert sorted(order_list) == list(
            range(len(a.shape))
        ), f"Invalid permutation order: {order_list}. Must be a valid permutation."
        return minitorch.Tensor(a._tensor.permute(*order_list), backend=a.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes backward pass for permute."""
        (order,) = ctx.saved_tensors

        inverse_order_storage = [0] * order._tensor.size
        for i in range(order._tensor.size):
            index = int(order._tensor._storage[i])
            inverse_order_storage[index] = i

        inverse_order = tensor(inverse_order_storage, backend=order.backend)

        grad_input = grad_output._tensor.permute(
            *[int(x) for x in inverse_order._tensor._storage]
        )

        zero_grad = zeros(order.shape, backend=order.backend)

        return minitorch.Tensor(grad_input, backend=grad_output.backend), zero_grad


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Computes forward pass for view."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes backward pass for view."""
        (original_shape,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage,
                original_shape,
                backend=grad_output.backend,
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a, zeros(a.shape, backend=a.backend))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(a, b)
        return a.f.matrix_multiply(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        a, b = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(b)),
            grad_output.f.matrix_multiply(transpose(a), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size shape.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size shape.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:Tensor : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape shape.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:Tensor : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference for a function."""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )