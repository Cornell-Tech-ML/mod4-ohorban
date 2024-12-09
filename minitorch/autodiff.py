from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Convert vals to a list so we can modify them
    vals_pos = list(vals)
    vals_neg = list(vals)

    # Modify the arg-th value by adding and subtracting epsilon
    vals_pos[arg] += epsilon
    vals_neg[arg] -= epsilon

    # Calculate the function values for both modified sets of arguments
    f_pos = f(*vals_pos)
    f_neg = f(*vals_neg)

    # Compute the central difference approximation
    return (f_pos - f_neg) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    """Protocol for a Variable in the computation graph.

    This defines the basic operations and properties that every
    Variable in the computation graph must implement.
    """

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative for the variable.

        Args:
        ----
            x (Any): The value to accumulate into the derivative.

        """
        ...

    @property
    def unique_id(self) -> int:
        """Returns a unique identifier for the variable.

        Returns
        -------
            int: A unique ID for this variable.

        """
        ...

    def is_leaf(self) -> bool:
        """Checks if the variable is a leaf node.

        Returns
        -------
            bool: True if the variable is a leaf (created by the user), False otherwise.

        """
        ...

    def is_constant(self) -> bool:
        """Checks if the variable is constant.

        Returns
        -------
            bool: True if the variable is constant and does not require gradient calculation.

        """
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables in the computation graph.

        Returns
        -------
            Iterable[Variable]: A collection of parent variables.

        """
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to propagate derivatives back through the graph.

        Args:
        ----
            d_output (Any): The derivative of the output with respect to some variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: A list of parent variables and their respective gradients.

        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable (the output of the computation).

    Returns:
    -------
        An iterable of non-constant variables in topological order starting from the right.

    """
    visited = set()
    topo_order = []

    def dfs(var: Variable) -> None:
        """Depth-first search to traverse the graph and build the topological order."""
        if var not in visited and not var.is_constant():
            visited.add(var)
            for parent in var.parents:
                dfs(parent)
            topo_order.append(var)

    dfs(variable)
    return reversed(topo_order)  # Return in reverse postorder for topological sorting


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Args:
    ----
        variable: The right-most variable (typically the output of a function) in the computation graph.
        deriv: The derivative of the output with respect to the variable, which is propagated backward.

    Returns:
    -------
        None: This function doesn't return a value, but it accumulates derivatives for each leaf variable.

    """
    # Topologically sort the variables (nodes)
    topo_order = topological_sort(variable)

    # Initialize a dictionary to store derivatives for each variable
    derivatives = {variable: deriv}

    # Traverse the nodes in reverse topological order
    for var in topo_order:
        if var.is_leaf():
            # If it's a leaf node, accumulate the derivative
            var.accumulate_derivative(derivatives.get(var, 0.0))
        else:
            # If it's an intermediate node, apply the chain rule and propagate the derivative
            d_output = derivatives.get(var, 0.0)
            for parent_var, local_derivative in var.chain_rule(d_output):
                if parent_var in derivatives:
                    derivatives[parent_var] += local_derivative
                else:
                    derivatives[parent_var] = local_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values for backpropagation."""
        return self.saved_values