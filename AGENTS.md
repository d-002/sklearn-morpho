# AGENTS.md - sklearn-morpho Developer Instructions

## Overview
`sklearn-morpho` is a scikit-learn compatible library written in Python (>=3.11) utilizing `uv` for dependency and environment management.
Refer to the provided `README.md` and `llms.txt` for further information on the package.

## Coding Standards
- Strictly adhere to standard scikit-learn Estimator conventions (`fit`, `predict`, `transform`) and the existing code organization.
- Type hints are required everywhere, run `mypy`, `ruff` as done in the GitHub ci workflows.
- Do not introduce new heavy dependencies without explicit justification.
- Ensure human users clearly understand the meaning and reasoning behind the additions made by LLMs and other automated tools to ensure maintainability.
