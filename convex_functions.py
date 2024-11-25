import numpy as np


class Phi:
    def __init__(self, functions):
        self.functions = functions

    def __call__(self, x):
        return -sum([np.log(-f(x)) for f in self.functions])

    def gradient(self, x):
        return -sum([f.gradient(x) / f(x) for f in self.functions])

    def hessian(self, x):
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

    def __init__(self, t, objective, phi):
        self.t = t
        self.objective = objective
        self.phi = phi

    def __call__(self, x):
        return self.objective(x) + self.phi(x) / self.t

    def gradient(self, x):
        return self.objective.gradient(x) + self.phi.gradient(x) / self.t

    def hessian(self, x):
        return self.objective.hessian(x) + self.phi.hessian(x) / self.t
