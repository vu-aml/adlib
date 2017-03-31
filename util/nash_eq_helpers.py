'''
Helper functions for Nash Equilibrium Learner and Adversary
'''

import numpy as np
from scipy.misc import derivative

# created util dir because both nash eq adversary and leaner need these functons.
# leaving them defined in the adversaries directory was causing a circular import error

def transform_instance(fv: np.array, a_adversary: np.array) -> np.array:
    fv = np.add(fv, a_adversary)
    return fv

def partial_derivative(func, var=0, point=[]):
    args = list(point)

    def wraps(x):
        args[var] = x
        return func(*args)

    return derivative(wraps, point[var], dx=1e-6)

def predict_instance(fv: np.array, a_learner: np.array) -> float:
    fv = np.dot(a_learner, fv)
    return fv

