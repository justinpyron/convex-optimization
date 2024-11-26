import cvxpy
import numpy as np
import streamlit as st

import convex_problem


class EqualityConstrainedObjective:
    def __call__(self, x: np.array) -> np.array:
        return x.dot(x) + np.exp(x).sum()

    def gradient(self, x: np.array) -> np.array:
        return 2 * x + np.exp(x)

    def hessian(self, x: np.array) -> np.array:
        return 2 * np.eye(x.shape[0]) + np.diag(np.exp(x))


class EqualityAndInequalityConstrainedObjective:
    """Represents the function f(x) = c^T x"""

    def __init__(self, c: np.array) -> None:
        self.c = c
        self.n = c.shape[0]

    def __call__(self, x: np.array) -> np.array:
        return self.c.dot(x)

    def gradient(self, x: np.array) -> np.array:
        return self.c

    def hessian(self, x: np.array) -> np.array:
        return np.zeros((self.n, self.n))


class EqualityAndInequalityConstrainedConstraint:
    """Represents the function f(x) = x^T x - 1"""

    def __call__(self, x: np.array) -> np.array:
        return x.dot(x) - 1

    def gradient(self, x: np.array) -> np.array:
        return 2 * x

    def hessian(self, x: np.array) -> np.array:
        return 2 * np.eye(x.shape[0])


def solve_quadratic_equality_constrained(
    num_variables: int,
    num_constraints: int,
) -> tuple[float, float]:
    # Problem data
    U = np.random.randn(num_variables, num_variables)
    P = U.T.dot(U)  # Creating P this way ensures it is positive semi-definite
    q = np.random.randn(num_variables)
    r = 0  # Does not affect optimal decision variable, so we can ignore
    A = np.random.randn(num_constraints, num_variables)
    b = A.dot(np.random.randn(num_variables))

    # Custom
    solution_custom, x = convex_problem.QuadraticEqualityConstrained(
        P, q, r, A, b
    ).solve()

    # CVXPY
    x = cvxpy.Variable(n)
    solution_cvxpy = cvxpy.Problem(
        objective=cvxpy.Minimize(0.5 * cvxpy.quad_form(x, P) + q.T @ x),
        constraints=[A @ x == b],
    ).solve()

    return solution_custom, solution_cvxpy


def solve_equality_constrained(
    num_variables: int,
    num_constraints: int,
) -> tuple[float, float]:
    # Problem data
    A = np.random.randn(num_constraints, num_variables)
    b = A.dot(np.random.randn(num_variables))

    # Custom
    function = EqualityConstrainedObjective()
    problem = convex_problem.EqualityConstrained(function, A, b)
    starting_point = np.linalg.lstsq(A, b, rcond=None)[0]
    solution_custom, x = problem.solve(starting_point)

    # CVXPY
    x = cvxpy.Variable(n)
    solution_cvxpy = cvxpy.Problem(
        objective=cvxpy.Minimize(cvxpy.norm(x) ** 2 + cvxpy.sum(cvxpy.exp(x))),
        constraints=[A @ x == b],
    ).solve()

    return solution_custom, solution_cvxpy


def solve_equality_and_inequality_constrained(
    num_variables: int,
    num_constraints: int,
) -> tuple[float, float]:
    # Problem data
    c = np.random.randint(-50, 50, num_variables)
    A = np.random.randn(num_constraints, num_variables)
    x_0 = np.random.uniform(size=num_variables) / np.sqrt(num_variables)
    b = A.dot(x_0)

    # Custom
    objective = EqualityAndInequalityConstrainedObjective(c)
    constraint_functions = [EqualityAndInequalityConstrainedConstraint()]
    solution_custom, x = convex_problem.EqualityAndInequalityConstrained(
        objective, constraint_functions, A, b
    ).solve(x_0)

    # CVXPY
    x = cvxpy.Variable(n)
    solution_cvxpy = cvxpy.Problem(
        objective=cvxpy.Minimize(c.T @ x),
        constraints=[cvxpy.norm(x) ** 2 <= 1, A @ x == b],
    ).solve()

    return solution_custom, solution_cvxpy


def report_results(
    solution_custom: float,
    solution_cvxpy: float,
) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.header("My solver")
        st.write("Optimal value = {:.7f}".format(solution_custom))
    with col2:
        st.header("CVXPY solver")
        st.write("Optimal value = {:.7f}".format(solution_cvxpy))
    st.info(
        "Relative error = {:.1E}".format(
            np.abs(solution_cvxpy - solution_custom) / solution_cvxpy
        ),
        icon="ℹ️",
    )


what_is_it = """
This app demos a [Convex Optimization](https://en.wikipedia.org/wiki/Convex_optimization) solver that I built from scratch. It uses the logarithmic barrier technique.

There are three tabs which demo three types of problems:
1. Quadratic Equality Constrained
2. Equality Constrained
3. Equality and Inequality Constrained

To validate my algorithm, I compare results obtained with the open-source [CVXPY library](https://www.cvxpy.org/).

To view the source code, see 👉 [GitHub](https://github.com/justinpyron/convex-optimization/blob/main/convex_problem.py).

For an explanation of the underlying math and algorithms, see 👉 [underlying math](https://github.com/justinpyron/convex-optimization/blob/main/underlying_math.pdf).
"""

st.set_page_config(page_title="Convex Opt", layout="centered", page_icon="🥇")
st.title("Convex Optimization 🥇")
with st.expander("What is it?"):
    st.markdown(what_is_it)

tab1, tab2, tab3 = st.tabs(
    [
        ":one: Quadratic Equality Constrained",
        ":two: Equality Constrained",
        ":three: Equality and Inequality Constrained",
    ]
)

with tab1:
    with st.form(key="tab1"):
        st.subheader("Problem description")
        st.markdown(
            """
            $$
            \\begin{equation*}
            \\begin{aligned}
            &\\text{minimize} & & \\frac{1}{2} x^T P x + q^T x + r \\\\
            &\\text{subject to} & & Ax = b
            \\end{aligned}
            \\end{equation*}
            $$
            """
        )
        st.subheader("Problem parameters")
        st.markdown(
            "Select the number of variables and equality constraints, "
            "then $P$, $q$, $r$, $A$, $b$ will be randomly generated."
        )
        n = st.slider("Number of variables", 1, 50, 25)
        m = st.slider("Number of equality constraints", 51, 100, 75)
        submitted = st.form_submit_button(
            "Solve", type="primary", use_container_width=True
        )
        if submitted:
            solution_custom, solution_cvxpy = solve_quadratic_equality_constrained(n, m)
            report_results(solution_custom, solution_cvxpy)

with tab2:
    with st.form(key="tab2"):
        st.subheader(
            "Problem description",
            help="This choice of objective function was arbitrary",
        )
        st.markdown(
            """
            $$
            \\begin{equation*}
            \\begin{aligned}
            &\\text{minimize} & & || x ||_2^2 + \\sum_{i=1}^n \\exp(x_i) \\\\
            &\\text{subject to} & & Ax = b
            \\end{aligned}
            \\end{equation*}
            $$
            """
        )
        st.subheader("Problem parameters")
        st.markdown(
            "Select the number of variables and equality constraints, "
            "then $A$ and $b$ will be randomly generated."
        )
        n = st.slider("Number of variables", 1, 50, 25)
        m = st.slider("Number of equality constraints", 51, 100, 75)
        submitted = st.form_submit_button(
            "Solve", type="primary", use_container_width=True
        )
        if submitted:
            solution_custom, solution_cvxpy = solve_equality_constrained(n, m)
            report_results(solution_custom, solution_cvxpy)

with tab3:
    with st.form(key="tab3"):
        st.subheader(
            "Problem description",
            help=(
                "This choice of objective function was arbitrary. "
                "It's a linear program whose solution is constrained "
                "to lie on a plane within the unit sphere."
            ),
        )
        st.markdown(
            """
            $$
            \\begin{equation*}
            \\begin{aligned}
            &\\text{minimize} & & c^T x \\\\
            &\\text{subject to} & & x^T x - 1 \\leq 0 \\\\
            & & & Ax = b \\\\
            \\end{aligned}
            \\end{equation*}
            $$
            """
        )
        st.subheader("Problem parameters")
        st.markdown(
            "Select the number of variables and equality constraints, "
            "then $c$, $A$, $b$ will be randomly generated."
        )
        n = st.slider("Number of variables", 1, 50, 25)
        m = st.slider("Number of equality constraints", 51, 100, 75)
        submitted = st.form_submit_button(
            "Solve", type="primary", use_container_width=True
        )
        if submitted:
            solution_custom, solution_cvxpy = solve_equality_and_inequality_constrained(
                n, m
            )
            report_results(solution_custom, solution_cvxpy)
