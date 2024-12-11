"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def mul(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def id(a: float) -> float:
    """Identity function."""
    return a


def neg(a: float) -> float:
    """Negate a number."""
    return -a


def lt(a: float, b: float) -> float:
    """Less than comparison."""
    if a < b:
        return 1.0
    else:
        return 0.0


def eq(a: float, b: float) -> float:
    """Equality comparison."""
    if a == b:
        return 1.0
    else:
        return 0.0


def max(a: float, b: float) -> float:
    """Maximum of two numbers."""
    if a > b:
        return a
    else:
        return b


def is_close(a: float, b: float) -> float:
    """Check if two numbers are close."""
    if abs(a - b) < 1e-2:
        return 1.0
    else:
        return 0.0


def sigmoid(a: float) -> float:
    """Sigmoid function."""
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        return math.exp(a) / (1.0 + math.exp(a))


def relu(a: float) -> float:
    """ReLU function."""
    if a > 0:
        return a
    else:
        return 0.0


def log(a: float) -> float:
    """Logarithm function."""
    return math.log(a)


def exp(a: float) -> float:
    """Exponential function."""
    return math.exp(a)


def log_back(a: float, b: float) -> float:
    """Returns the derivative of log(a) with respect to a, multiplied by b. Assuming it's for backpropagation."""
    return b / a


def inv(a: float) -> float:
    """Inverse function."""
    return 1.0 / a


def inv_back(a: float, b: float) -> float:
    """Inverse function."""
    return -b / (a * a)


def relu_back(grad_output: float, input: float) -> float:
    """ReLU function."""
    return grad_output if input > 0 else 0.0


# Small practice library of elementary higher-order functions.
def map(fn: Callable[[float], float], a: Iterable[float]) -> Iterable[float]:
    """Map a function over a list."""
    return [fn(x) for x in a]


def zipWith(
    fn: Callable[[float, float], float], a: Iterable[float], b: Iterable[float]
) -> Iterable[float]:
    """Zip two lists together with a function."""
    return [fn(x, y) for x, y in zip(a, b)]


def reduce(
    fn: Callable[[float, float], float], a: Iterable[float], init: float
) -> float:
    """Reduce a list with a function."""
    acc = init
    for x in a:
        acc = fn(acc, x)
    return acc


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg, a)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Add two lists."""
    return zipWith(add, a, b)


def sum(a: Iterable[float]) -> float:
    """Sum a list."""
    return reduce(add, a, 0.0)


def prod(a: Iterable[float]) -> float:
    """Product of a list."""
    return reduce(mul, a, 1.0)
