import numpy as np


class Phi:
    def __init__(self, functions: list[callable]) -> None:
        self.functions = functions

    def __call__(self, x: np.array) -> float:
        return -sum([np.log(-f(x)) for f in self.functions])

    def gradient(self, x: np.array) -> np.array:
        return -sum([f.gradient(x) / f(x) for f in self.functions])

    def hessian(self, x: np.array) -> np.array:
        a = sum(
            [
                np.outer(f.gradient(x), f.gradient(x)) / np.square(f(x))
                for f in self.functions
            ]
        )
        b = -sum([f.hessian(x) / f(x) for f in self.functions])
        return a + b


class Omega:
    """
    Omega is the function defined by

    omega(x) = objective(x) + (1/t) * phi(x)

    where t > 0 is fixed.
    """

    def __init__(
        self,
        t: float,
        objective: callable,
        phi: Phi,
    ) -> None:
        self.t = t
        self.objective = objective
        self.phi = phi

    def __call__(self, x: np.array) -> float:
        return self.objective(x) + self.phi(x) / self.t

    def gradient(self, x: np.array) -> np.array:
        return self.objective.gradient(x) + self.phi.gradient(x) / self.t

    def hessian(self, x: np.array) -> np.array:
        return self.objective.hessian(x) + self.phi.hessian(x) / self.t
