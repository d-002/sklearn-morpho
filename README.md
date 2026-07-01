# sklearn-morpho

Scikit-learn estimator toolbox for morphological perceptrons.

![](https://img.shields.io/pypi/v/sklearn-morpho?style=flat-square)
![](https://img.shields.io/aur/version/python-sklearn-morpho?style=flat-square)
![](https://img.shields.io/github/actions/workflow/status/d-002/sklearn-morpho/ci.yml?style=flat-square)

Current features:

- Scikit-learn estimators:
  - Linear Dilation-Erosion Perceptron (l-DEP)
  - Reduced Dilation-Erosion Perceptron (r-DEP)
  - Simple Dilation and Erosion Morphological Perceptrons
- Modular wrapper for DCCP optimization tasks with `cvxpy`

File tree:

- `src/sklearn_morpho`: contains the source code and a testsuite in its `tests`
  subdirectory.
- `tests`: pytest testsuite.
- `testing`: standalone files that use this library.
  They are used internally for testing, but they are not tests nor part of the
  library.
- `MREs`: standalone jupyter notebooks to showcase some of this library's
  features.

## Getting started

Take a look at the Jupyter code examples in the
[MREs](https://github.com/d-002/sklearn-morpho/tree/master/MREs) directory.

## Running the project

Install Python 3 and hatch.
Then run one of these commands:

- `hatch run jupyter lab` to run the Jupyter notebooks
- `hatch run pytest` for tests
- `hatch shell` to run testing files like `testing/display_boundary.py` in the
  right environment.

Special note for the estimators comparison testing files: they are split in two
files to avoid training the estimators every time one wants to view the results.

## Contributing

As the project is open source, any help is greatly appreciated!

To keep code clean, this repository uses a CI/CD pipeline with tests and an
enforced coding style, which can all be found in the project's configuration
files.

Please adhere to these rules when contributing.

Regarding AI pull requests, they are generally discouraged as you should be able
to understand and help maintaining any features you add to this repo.

## For Arch users

The Python package is available in the
[Arch User Repository](https://aur.archlinux.org) as `python-sklearn-morpho`, as
are all its dependencies not already in the official packages, except for
`dccp`.

Since the latter is an optional dependency, you can either:

- Download `dccp` through pip or use `pip install sklearn-morpho[dccp]` in a
  Python virtual environment.
- Download `python-sklearn-morpho` from the AUR and use
  `pip install dccp --user`.
