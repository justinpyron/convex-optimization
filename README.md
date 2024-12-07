# convex-optimization
From-scratch convex optimization solver.

It solves three types of convex problems using an interior point method:
1. Quadratic Equality Constrained
2. Equality Constrained
3. Equality and Inequality Constrained

See `underlying_math.pdf` for a comprehensive overview of the algorithm used by the solver.

# Project Organization
```
├── README.md                <- Overview
├── app.py                   <- Streamlit web app frontend
├── convex_problem.py        <- Class representing three types of convex problems
├── convex_functions.py      <- Building-block functions of the interior-point method
├── underlying_math.pdf      <- Overview of Convex Optimization math and algorithms
├── pyproject.toml           <- Poetry config specifying Python environment dependencies
├── poetry.lock              <- Locked dependencies to ensure consistent installs
├── .pre-commit-config.yaml  <- Linting configs
```

# Installation
This project uses [Poetry](https://python-poetry.org/docs/) to manage its Python environment.

1. Install Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies
```
poetry install
```

# Usage
A Streamlit web app is the frontend for interacting with the solver.

The app can be accessed at https://convex-optimization-from-scratch.streamlit.app.

Alternatively, the app can be run locally with
```
poetry run streamlit run app.py
```
