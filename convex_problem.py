import numpy as np

import convex_functions


class QuadraticEqualityConstrained:
    """
    Solves equality-constrained quadratic optimization problems of the form

    minimize      0.5 x^T P x + q^T x + r
    subject to    Ax = b

    where P is positive semi-definite.
    """

    def __init__(
        self,
        P: np.array,
        q: np.array,
        r: float,
        A: np.array,
        b: np.array,
    ) -> None:
        self.P = P
        self.q = q
        self.r = r
        self.A = A
        self.b = b
        self.n = P.shape[0]
        self.m = A.shape[0]

    def solve(self) -> tuple[np.float64, np.array]:
        """
        Solves the optimization problem.

        Returns
        -------
        solution : The optimal value of the objective function
        x : The point at which the objective function is optimal
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
        x = solution[: self.n]  # Extract primal variable from KKT system
        solution = 0.5 * x.dot(self.P.dot(x)) + self.q.dot(x) + self.r
        return solution, x


class EqualityConstrained:
    """
    Solves equality-constrained optimization problems of the form

    minimize      objective_function(x)
    subject to    Ax = b

    where objective_function is convex.
    """

    def __init__(
        self,
        objective_function: callable,
        A: np.array,
        b: np.array,
    ) -> None:
        self.f = objective_function
        self.A = A
        self.b = b
        # TODO Implement Phase I method to compute a feasible starting point, or determine that one does not exist

    def compute_step(self, x: np.array) -> np.array:
        qec_problem = QuadraticEqualityConstrained(
            P=self.f.hessian(x),
            q=self.f.gradient(x),
            r=0,
            A=self.A,
            b=self.b - self.A.dot(x),
        )
        _, step = qec_problem.solve()
        return step

    def solve(
        self,
        starting_point: np.array,
        tol: float = 1e-5,
    ) -> tuple[np.float64, np.array]:
        """
        Solves the optimization problem.

        Arguments
        ---------
        starting_point : The initial point from which optimization algorithm begins. Must be feasible.

        Returns
        -------
        solution : The optimal value of the objective function
        x : The point at which the objective function is optimal
        """
        x = starting_point
        change = -tol - 1  # Subtract 1 to ensure while loop starts
        while change < -tol:
            new_x = x + self.compute_step(x)
            change = self.f(new_x) - self.f(x)
            x = new_x
        solution = self.f(x)
        return solution, x


class EqualityAndInequalityConstrained:
    """
    Solves equality and inequality-constrained optimization problems of the form

    minimize      objective_function(x)
    subject to    constraint_functions[i](x) <= 0, i=1,...,n
                  Ax = b

    where objective_function and constraint_functions[i] for i=1,...,n are convex.
    """

    def __init__(
        self,
        objective_function: callable,
        constraint_functions: list[callable],
        A: np.array,
        b: np.array,
    ) -> None:
        """
        Arguments
        ---------
        objective_function : The objective function
        constraint_functions : List of constraint functions
        A : The A matrix from linear equality constraints
        b : The b vector from linear equality constraints
        """
        self.objective_function = objective_function
        self.log_barrier = convex_functions.LogBarrier(constraint_functions)
        self.A = A
        self.b = b

    def solve(
        self,
        starting_point: np.array,
        tol: float = 1e-4,
    ) -> tuple[np.float64, np.array]:
        """
        Solves the optimization problem.

        Arguments
        ---------
        starting_point : The initial point from which optimization algorithm begins. Must be feasible.

        Returns
        -------
        solution : The optimal value of the objective function
        x : The point at which the objective function is optimal
        """
        max_t = 1e5
        growth_rate = 1.5
        min_iterations = 4
        x = starting_point.copy()
        solution = self.objective_function(x)
        t = 1e-6
        num_iterations = 0
        while t < max_t:
            approx_objective = convex_functions.ApproximatedObjective(
                t, self.objective_function, self.log_barrier
            )
            ec_problem = EqualityConstrained(approx_objective, self.A, self.b)
            _, new_x = ec_problem.solve(x)
            change = self.objective_function(new_x) - self.objective_function(x)
            x = new_x
            t *= growth_rate
            num_iterations += 1
            if change > -tol and num_iterations >= min_iterations:
                break
        solution = self.objective_function(x)
        return solution, x
