# Contributing to pyMechT

Thanks for contributing!

## Development installation

To install as a devloper, it is recommended to fork from the repo and clone this fork locally.
### *Step 1 (optional): Create a virtual environment*

To create an environment in Anaconda, execute:
```sh
conda create -n pymecht
```

To activate this virtual environment, execute:
```sh
conda activate pymecht
```
This is an option, but recommended step. There are other options for create and managing environments (such as venv or virtualenv)

### *Step 2: Install via pip*
To install as a devloper, it is recommended to fork from the repo and clone this fork locally.
### *Step 2.1 Fork from ankushaggarwal/pymecht*
To fork a branch, head to the [Github repository](https://github.com/ankushaggarwal/pymecht) and click the fork button in the top right-hand corner.
### *Step 2.2 Clone the forked repo*
To clone this repo locally, use the
```sh
git clone <repo-address>
```
where `<repo-address>` can be replaced by either the https or ssh addresses of the forked repo.

### *Step 2.3 Install developer version of pyMechT*
To install a developer version of pyMechT, navigate to the locally cloned repo and execute:
```sh
python setup.py develop
```
An editable version of pyMechT is now installed. All local changes to the cloned source code files will be reflected when pyMechT is imported.


## Our Development Process

### Docstrings
We use [NumPy/SciPy style sphinx docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html).


### Unit Tests

We use python's `pytest` to run unit tests:
```bash
python -m pytest
```

- To run all unit tests, run `python -m pytest` or `pytest` in the project directory 

See [pytest documentation](https://docs.pytest.org/en/stable/how-to/usage.html) for running the tests on specific folders/files/functions.

### Documentation

pyMechT uses sphinx to generate documentation, and ReadTheDocs to host documentation.
To build the documentation locally, ensure that sphinx and its plugins are properly installed. 
Then run:

```bash
cd docs
make html
```

To view the documentation, run
```
cd build/html
python -m http.server 8000
```

The documentation will be available at http://localhost:8000.
You will have to rerun the `make html` command every time you wish to update the docs.

## Pull Requests
We greatly appreciate PRs! To minimze back-and-forward communication, please ensure that your PR includes the following:

1. **Code changes.** (the bug fix/new feature/updated documentation/etc.)
1. **Unit tests.** If you are updating any code, you should add an appropraite unit test.
   - If you are fixing a bug, make sure that there's a new unit test that catches the bug.
     (I.e., there should be a new unit test that fails before your bug fix, but passes after your bug fix.
     This ensures that we don't have any accidental regressions in the code base.)
   - If you are adding a new feature, you should add unit tests for this new feature.
1. **Documentation.** Any new objects/methods should have [appropriate docstrings](#docstrings).
   - If you are adding a new object, **please ensure that it appears in the documentation.**
     You may have to add the object to the appropriate file in [docs/source](https://github.com/ankushaggarwal/pymecht/tree/master/docs/source/)
1. **Example notebooks.** Any major new functionality, tutorials, or examples should have an example jupyter notebook.
   - If you are adding a new notebook, **please ensure that it appears in the documentation.**
     You may have to add the object to the appropriate file in [docs/source](https://github.com/ankushaggarwal/pymecht/tree/master/docs/source/index.rst).

Before submitting a PR, ensure the following:
1. **Code is proprerly formatted and linted.** 
1. **Unit tests pass.** See [the unit tests section](#unit-tests) for more info.
1. **Documentation renders correctly without warnings.** [Build the documentation locally](#documentation) to ensure that your new class/docstrings are rendered correctly. Ensure that sphinx can build the documentation without warnings.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

We accept the following types of issues:
- Bug reports
- Requests for documentation/examples
- Feature requests
- Opportuntities to refactor code
- Performance issues (speed, memory, etc.)

## License

By contributing to pyMechT, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
