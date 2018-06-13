# A set of common functions
# Matthew Sedam. 2018.

import math
import numpy as np


def fuzz_matrix(matrix: np.ndarray):
    """
    Add to every entry of matrix some noise to make it non-singular.
    :param matrix: the matrix - 2 dimensional
    """

    m = matrix.tolist()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            m[i][j] += abs(np.random.normal(0, 0.00001))

    return np.array(m)


def logistic_function(x):
    """
    :param x: x
    :return: the logistic function of x
    """

    return 1 / (1 + math.exp(-1 * x))
