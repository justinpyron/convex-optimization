import numpy as np


class LogBarrier:
    """
    Represents a Logarithmic Barrier function.

    Consider a problem of the form

    minimize      objective_function(x)
    subject to    constraint_functions[i](x) <= 0, i=1,...,n
                  Ax = b

    where objective_function and constraint_functions[i] for i=1,...,n are convex.

    Then, the Logarithmic Barrier function phi is given by

    phi(x) = - sum_{i=1}^n log( -constraint_functions[i](x) )
    """

    def __init__(self, constraint_functions: list[callable]) -> None:
        self.constraint_functions = constraint_functions

    def __call__(self, x: np.array) -> float:
        return -sum([np.log(-f(x)) for f in self.constraint_functions])

    def gradient(self, x: np.array) -> np.array:
        return -sum([f.gradient(x) / f(x) for f in self.constraint_functions])

    def hessian(self, x: np.array) -> np.array:
        a = sum(
            [
                np.outer(f.gradient(x), f.gradient(x)) / np.square(f(x))
                for f in self.constraint_functions
            ]
        )
        b = -sum([f.hessian(x) / f(x) for f in self.constraint_functions])
        return a + b


class ApproximatedObjective:
    """
    Represents an approximation of an objective function where inequality
    constraints are implicitly incorporated via a logarithmic barrier function.

    The ApproximatedObjective g is given by

    g(x) = objective_function(x) + (1/t) * LogBarrier(x)

    where t > 0 is fixed and LogBarrier is the associated Logarithmic Barrier function.
    Larger values of t result in closer approximations of the objective function.
    """

    def __init__(
        self,
        t: float,
        objective: callable,
        log_barrier: LogBarrier,
    ) -> None:
        self.t = t
        self.objective = objective
        self.log_barrier = log_barrier

    def __call__(self, x: np.array) -> float:
        return self.objective(x) + self.log_barrier(x) / self.t

    def gradient(self, x: np.array) -> np.array:
        return self.objective.gradient(x) + self.log_barrier.gradient(x) / self.t

    def hessian(self, x: np.array) -> np.array:
        return self.objective.hessian(x) + self.log_barrier.hessian(x) / self.t
