![PyPI - Version](https://img.shields.io/pypi/v/pymecht) ![Build Status](https://github.com/ankushaggarwal/pymecht/actions/workflows/ci-tests.yml/badge.svg) ![Python versions](https://img.shields.io/badge/python-3.8%2B-blue.svg) ![GitHub](https://img.shields.io/github/license/ankushaggarwal/pymecht)

# PYMECHT

This PYthon-based repository is for the MECHanics of Tissues. The focus is on flexibility of adding new material models (referred to as *MatModels*) and varying their parameters.The ethos of PYMECHT is to create a virtual experimental setup. That is, a *sample* is created of which there are the following options:
* OPTION1
* OPTION2
* OPTION3
* ...

The MatModel is applied to the sample, such that the parameters encompass both the material parameters and those which define the corresponding sample setup (an example being the dimensions of the specimen). Finally, the samples can be simulated in either *displacement-controlled* or *force-controlled* loading.

Required dependencies are:
* matplotlib
* numpy
* pandas
* pyDOE
* scipy
* torch
* sympy
* tqdm

# Installation

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

<details>
<summary>User</summary>

Pymecht can be installed directly from PyPI via pip by using:
```sh
pip3 install pymecht
```

</details>

<details>
<summary>Developer</summary>
To install as a devloper, it is recommended to fork from the repo and clone this fork locally.

### *Step 2.1 Fork from ankushaggarwal/pymecht*
To fork a branch, head to the Github repo https://github.com/ankushaggarwal/pymecht and click the fork button in the top right-hand corner.
### *Step 2.2 Clone the forked repo*
To clone this repo locally, use the
```sh
git clone <repo-address>
```
where `<repo-address>` can be replaced by either the https or ssh addresses of the forked repo.

### *Step 2.3 Install pymecht as editable*
To install an editable version of pymecht, navigate to the locally cloned repo and execute:
```sh
pip3 install -e .
```
An editable version of pymecht is now installed. All local changes to the cloned source code files will be reflected when pymecht is imported.

</details>

### *Step 3: Check installation*

Ensure that pymecht has been installed by executing:
```sh
pip3 list
```
The package and version should be visible in the resulting list.

# Contributing to pymecht

To contribute to the pymecht framework, install pymecht using the developer options. All changes should be made to your forked repo. If there is a new feature or bug fix, raise a pull request. In the case that an additional feature is added, a corresponding example and test should be written in the respective python scripts.
