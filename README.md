# PYMECHT
This PYthon-based repository is for the MECHanics of Tissues. 
The focus is on flexibility of adding new constitutive models and varying their parameters.

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

Pymecht can be installed directly from Github using:
```sh
pip install git+https://github.com/ankushaggarwal/pymecht.git
```
> **Note**
> A personal access token may require to be setup in order to install via https. See https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token.

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
pip install -e .
```
An editable version of pymecht is now installed. All local changes to the cloned source code files will be reflected when pymecht is imported.

</details>

### *Step 3: Check installation*

Ensure that pymecht has been installed by executing:
```sh
pip list
```
The package and version should be visible in the resulting list.

# Contributing to pymecht

To contribute to the pymecht framework, install pymecht using the developer options. All changes should be made to your forked repo. If there is a new feature or bug fix, raise a pull request. In the case that an additional feature is added, a corresponding example and test should be written in the respective python scripts.
