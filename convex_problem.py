import numpy as np

import convex_functions


class QuadraticEqualityConstrained:
    """
    Solves equality-constrained quadratic optimization problems of the form

    minimize      0.5 x^T P x + q^T x + r
    subject to    Ax = b
    """

    def __init__(self, P, q, r, A, b):
        self.P = P
        self.q = q
        self.r = r
        self.A = A
        self.b = b
        self.n = P.shape[0]
        self.m = A.shape[0]

    def solve(self):
        """
        Solve the optimization problem

        Returns
        -------
        solution : float
            The optimal value of the objective function
        x : np.array
            The point at which the objective function is optimal
        """
        RHS = np.concatenate(
            [
                np.concatenate([self.P, self.A.T], axis=1),
                np.concatenate([self.A, np.zeros((self.m, self.m))], axis=1),
            ],
            axis=0,
        )
        LHS = np.concatenate([-self.q, self.b])

        solution = np.linalg.lstsq(RHS, LHS, rcond=None)[0]
        x = solution[: self.n]  # extract primal variable
        solution = 0.5 * x.dot(self.P.dot(x)) + self.q.dot(x) + self.r

        return solution, x


class EqualityConstrained:
    """
    Solves equality-constrained optimization problems of the form

    minimize      f(x)
    subject to    Ax = b

    where f is convex.
    """

    def __init__(self, objective_function, A, b):
        self.f = objective_function
        self.A = A
        self.b = b

    # TODO implement Phase I method to compute a feasible starting
    # point, or determine that one does not exist

    def solve(self, starting_point, tol=1e-5):
        """
        Solve the optimization problem

        Arguments
        ---------
        starting_point : np.array
            The initial point from which optimization algorithm
            begins. Must be feasible.

        Returns
        -------
        solution : float
            The optimal value of the objective function
        x : np.array
            The point at which the objective function is optimal
        """
        x = starting_point
        change = -tol - 1  # ensure while loop starts
        i = 0
        while change < -tol:
            new_x = x + self.compute_step(x)
            change = self.f(new_x) - self.f(x)
            x = new_x
        solution = self.f(x)

        return solution, x

    def compute_step(self, x):
        problem = QuadraticEqualityConstrained(
            P=self.f.hessian(x),
            q=self.f.gradient(x),
            r=0,
            A=self.A,
            b=self.b - self.A.dot(x),
        )
        _, step = problem.solve()
        return step


class InequalityConstrained:
    """
    Solves inequality-constrained optimization problems of the form

    minimize      f_0(x)
    subject to    f_i(x) <= 0, i=1,...,n

    where f_i, i=0,...,n are convex.
    """

    def __init__(self):
        pass

    # TODO implement this class


class EqualityAndInequalityConstrained:
    """
    Solves equality and inequality-constrained optimization problems of the form

    minimize      f_0(x)
    subject to    f_i(x) <= 0, i=1,...,n
                  Ax = b

    where f_i, i=0,...,n are convex.
    """

    def __init__(self, objective, constraint_functions, A, b):
        """
        Arguments
        ---------
        objective : object representing a function
            The objective function
        constraint_functions : [object representing a function]
            List of constraint functions
        A : np.array
            A matrix from linear equality constraints
        b : np.array
            b vector from linear equality constraints
        """
        self.objective = objective
        self.phi = convex_functions.Phi(constraint_functions)
        self.A = A
        self.b = b

    def solve(self, starting_point, tol=1e-4):
        """
        Solve the optimization problem

        Arguments
        ---------
        starting_point : np.array
            The initial point from which optimization algorithm
            begins. Must be feasible.

        Returns
        -------
        solution : float
            The optimal value of the objective function
        x : np.array
            The point at which the objective function is optimal
        """
        x = starting_point.copy()
        solution = self.objective(x)

        t = 1e-6
        i = 0
        while t < 1e5:
            omega = convex_functions.Omega(t, self.objective, self.phi)
            problem = EqualityConstrained(omega, self.A, self.b)
            _, new_x = problem.solve(x)
            change = self.objective(new_x) - self.objective(x)
            x = new_x
            t *= 1.5
            i += 1
            if change > -tol and i >= 4:  # arbitrary min iteration threshold
                break
        solution = self.objective(x)

        return solution, x
