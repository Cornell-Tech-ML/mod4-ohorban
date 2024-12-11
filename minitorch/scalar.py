from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    Eq,
    Lt,
    Add,
    Sub,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    Relu,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """True if this variable is a constant (no derivative)"""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the input variables of the last function that created this variable."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Compute the chain rule for the derivatives of the inputs."""
        # Retrieve the history
        h = self.history
        assert h is not None, "No history found for this scalar."
        assert h.last_fn is not None, "No last function found for this scalar."
        assert h.ctx is not None, "No context found for this scalar."

        # Get the last function and the saved inputs for the current scalar
        last_fn = h.last_fn
        inputs = h.inputs
        ctx = h.ctx

        # Compute the local derivatives using the backward function of the last function
        local_derivatives = last_fn._backward(ctx, d_output)

        # Pair each local derivative with the corresponding input variable
        return [
            (input_var, local_derivatives[i])
            for i, input_var in enumerate(inputs)
            if not input_var.is_constant()
        ]

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)

    def __add__(self, other: ScalarLike) -> Scalar:
        return Add.apply(self, other)

    def __sub__(self, other: ScalarLike) -> Scalar:
        return Sub.apply(self, other)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __mul__(self, other: ScalarLike) -> Scalar:
        return Mul.apply(self, other)

    def __lt__(self, other: ScalarLike) -> Scalar:
        return Lt.apply(self, other)

    def __eq__(self, other: ScalarLike) -> Scalar:
        return Eq.apply(self, other)

    def log(self) -> Scalar:
        """Return the natural logarithm of the scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Return the exponential of the scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Return the sigmoid of the scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Return the ReLU of the scalar."""
        return Relu.apply(self)

    def __hash__(self):
        """Use unique_id for hashing to allow Scalar objects to be used in sets."""
        return hash(self.unique_id)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks the correctness of autodiff by comparing the computed derivatives with central difference approximations.

    Asserts False if the derivative calculated by autodiff differs significantly from the central difference approximation.

    Args:
    ----
        f: A function that takes n-scalars as input and returns a single scalar as output.
           This is the function whose derivatives we want to check.
        *scalars: A variable number of Scalar objects that are the inputs to the function `f`.
                  These scalars will be used to compute and verify the derivatives.

    Returns:
    -------
        None: The function raises an assertion error if the autodiff derivatives do not match
              the central difference approximation.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
