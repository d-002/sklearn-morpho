# sklearn-morpho

scikit-learn estimator toolbox for morphological perceptrons.

> [!WARNING]  
> Some features of this repository use the `dccp` module.
>
> As of writing this (June 2026) the PyPI package has not been updated since
> 2023 ([this commit](https://github.com/cvxgpr/dccp/commit/4322809)).
> This means that currently the `dccp` module uses `cvxpy.reshape` with soon to
> be outdated parameters.
> When using this solver, warnings are currently displayed but the correct
> behavior is observed; however these warnings are projected to be removed in
> future `cvxpy` versions, which may create silent errors during training.
>
> I will leave this notice here until the `dccp` library's maintainer team
> decides to update the package.
> Please let me know if that is the case and I did not remove the notice.
>
> Since the PyPI package is affected, the hatch commands presented below are as
> well.
> To prevent this issue manually, you can clone the dccp repo into the hatch
> python venv in `.venv`.

Current features:

- Scikit-learn estimators:
  - Linear Dilation-Erosion Perceptron (l-DEP)
  - Reduced Dilation-Erosion Perceptron (r-DEP)
  - Simple dilation and erosion morphological perceptrons
- Modular wrapper for DCCP optimization tasks with `cvxpy`

File tree:

- `src/sklearn_morpho`: contains the source code and a testsuite in `.../tests`
- `testing`: standalone files that use this library, may contain tests but they
  are not designed to be run as a CI testsuite for example.
- `MREs`: standalone jupyter notebooks to showcase some of this library's
  features.

## Getting started

Take a look at the Jupyter code examples in the
[`MREs`](https://github.com/d-002/sklearn-morpho/tree/master/MREs) directory.

## Running the project

Install Python 3 and hatch.
Then run one of these commands:

- `hatch run jupyter lab` to run the Jupyter notebooks
- `hatch run pytest` for tests
- `hatch shell` to run testing files like `testing/display_boundary.py` in the
  right environment.

Special note for the estimators comparison testing files: they are split in two
files to avoid training the estimators every time one wants to view the results.
