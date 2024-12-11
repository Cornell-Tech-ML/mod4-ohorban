"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """History stores the history of Function operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets whether the tensor requires a gradient."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Returns whether the tensor requires a gradient."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of backward
        is a different size than the input of forward.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of other with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Create a new zero tensor with the same shape as this tensor."""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add val to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no last_fn)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is constant (no gradient required)"""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables in the computation graph."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to propagate derivatives back through the graph."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        result = []
        for inp, d_in in zip(h.inputs, x):
            if d_in is not None:
                result.append((inp, inp.expand(self._ensure_tensor(d_in))))
        return result

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Backpropagate the gradient through the computation graph."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    def view(self, *shape: int, dim: Optional[int] = None) -> Tensor:
        """Reshapes the tensor using the View function."""
        if dim is None:
            return View.apply(self.contiguous(), tensor(shape))
        else:
            shape_list = list(self.shape)
            shape_list[dim] = shape[0]
            return View.apply(self.contiguous(), tensor(shape_list))

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions in the tensor."""
        return len(self.shape)

    def all(self, dim: Optional[Tensor] = None) -> Tensor:
        """Returns whether all elements are true."""
        if dim is not None:
            return All.apply(self, self._ensure_tensor(dim))
        else:
            return All.apply(self.view(self.size), self._ensure_tensor(0))

    def add(self, other: TensorLike) -> Tensor:
        """Add two tensors element-wise."""
        return Add.apply(self, self._ensure_tensor(other))

    def sub(self, other: TensorLike) -> Tensor:
        """Subtract two tensors element-wise."""
        return self.add(self._ensure_tensor(other).neg())

    def mul(self, other: TensorLike) -> Tensor:
        """Multiply two tensors element-wise."""
        return Mul.apply(self, self._ensure_tensor(other))

    def neg(self) -> Tensor:
        """Negate the tensor element-wise."""
        return Neg.apply(self)

    def lt(self, other: TensorLike) -> Tensor:
        """Implements element-wise 'less than' comparison."""
        return LT.apply(self, self._ensure_tensor(other))

    def eq(self, other: TensorLike) -> Tensor:
        """Implements element-wise equality between two tensors."""
        return EQ.apply(self, self._ensure_tensor(other))

    def sigmoid(self) -> Tensor:
        """Implements the sigmoid function."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Implements the ReLU function."""
        assert self.f is not None, "Backend is not initialized!"
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Implements the natural logarithm."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Implements the exponential function."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Sum the tensor over a dimension."""
        if dim is None:
            return self.view(int(operators.prod(self.shape))).sum(0)
        else:
            dim_tensor = Tensor.make([dim], (1,), backend=self.backend)
            return Sum.apply(self, dim_tensor)

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Calculate the mean over a dimension."""
        if dim is None:
            return self.sum(dim) / self.size
        else:
            return self.sum(dim) / self.shape[dim]

    def permute(self, *order: int) -> Tensor:
        """Permute the dimensions of the tensor."""
        order_tensor = Tensor.make(list(order), (len(order),), backend=self.backend)
        return Permute.apply(self, order_tensor)

    def is_close(self, other: TensorLike) -> Tensor:
        """Implements element-wise closeness between two tensors."""
        return IsClose.apply(self, self._ensure_tensor(other))

    def __add__(self, other: TensorLike) -> Tensor:
        return self.add(other)

    def __radd__(self, other: TensorLike) -> Tensor:
        return self.add(other)

    def __sub__(self, other: TensorLike) -> Tensor:
        return self.sub(other)

    def __rsub__(self, other: TensorLike) -> Tensor:
        return self._ensure_tensor(other).sub(self)

    def __mul__(self, other: TensorLike) -> Tensor:
        return self.mul(other)

    def __rmul__(self, other: TensorLike) -> Tensor:
        return self.mul(other)

    def __neg__(self) -> Tensor:
        return self.neg()

    def __eq__(self, other: TensorLike) -> Tensor:
        """Implements element-wise equality between two tensors."""
        return EQ.apply(self, self._ensure_tensor(other))

    def __lt__(self, other: TensorLike) -> Tensor:
        """Implements the 'less than' operator."""
        return LT.apply(self, self._ensure_tensor(other))

    def __gt__(self, other: TensorLike) -> Tensor:
        """Implements the 'greater than' operator by reversing operands."""
        return LT.apply(self._ensure_tensor(other), self)

    def zero_grad_(self) -> None:
        """Clears the gradient of the tensor."""
        self.grad = None

    def __hash__(self):
        """Hash the tensor."""
        return id(self)
