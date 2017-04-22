# AML

[![Build Status](https://travis-ci.com/vu-aml/adlib.svg?token=UQbyf6hz4VVL6qGppwSf&branch=master)](https://travis-ci.com/vu-aml/adlib)


data_reader/data is in .gitignore to speed up git. If you need to make a change from one of those
files, use git add -f 

## Unit Testing
We use py.test for unit testing. Install with `pip install -U pytest`.

To run tests, run `python -m pytest tests/<path to test>` from the root directory.


## Docs
To serve the docs locally, you may have to install Sphinx with: `pip install Sphinx`
- First, try to open index.html with your browser, if that doesn't work, you make have to 
 make the html. 
  * In that case, navigate to `docs/` and run: `make html`.


## Installation
### Dependencies
* Python3 
* SciPy
* NumPy
* Matplotlib
* Scikit-learn
* CVXPY
* CVXOPT (optional as a CVXPY solver)
* Jupyter Notebook (optional for notebook demo) 
