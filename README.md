# sklearn-morpho

> [!WARNING]  
> This repo is still a work in progress.
> More features, improved documentation and examples are still to come.

Scikit-learn estimator toolbox for morphological perceptrons.

Current features:

- Linear Dilation-Erosion Perceptron as a scikit-learn estimator
- Modular wrapper for DCP optimization tasks with `cvxpy`

File tree:
- `src/sklearn_morpho`: contains the source code and a testsuite in `.../tests`
- `testing`: standalone files that use this library, may contain tests but they
  are not designed to be run as a CI testsuite for example.
- `MREs`: standalone jupyter notebooks to showcase some of this library's
  features.

## Getting started

Take a look at the Jupyter code examples in the `MREs` directory.

## Running the project

Install Python 3 and hatch.
Then run one of these commands:

- `hatch run jupyter lab` to run the Jupyter notebooks
- `hatch run pytest` for tests
- `hatch shell` to run testing files like `testing/display_boundary.py` in the
  right environment.

Special note for the estimators comparison testing files: they are split in two
files to avoid training the estimators every time one wants to view the results.
