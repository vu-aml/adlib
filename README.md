# Adversarial Machine Learning Library (AML)
### Computational Economics Research Lab at Vanderbilt University

[![Travis CI](https://travis-ci.org/vu-aml/adlib.svg?branch=master)](https://travis-ci.org/vu-aml/adlib)

Game-theoretic adversarial machine learning library providing a set of learner and adversary modules.

### Installation
To install the dependencies for `adlib` do `pip install -r requirements.txt`. See below for a list of dependencies.
To install `adlib`, run `python3 setup.py install`. For development, do `python3 setup.py develop`.

#### Dependencies
* Python3 
* SciPy
* NumPy
* Matplotlib
* Scikit-learn
* CVXPY (0.4-0.4.11 version)
* Pathos
* Pandas
* CVXOPT (optional as a CVXPY solver)
* Jupyter Notebook (optional for notebook demo)
* Py.test (optional for testing)

### License
Copyright 2016-2018 Computational Economics Research Lab. Released under the MIT License. See `LICENSE` for details.

### Note
`data_reader/data` is in `.gitignore` to speed up `git`. If you need to make a change from one of those
files, use `git add -f $FILE$`.
