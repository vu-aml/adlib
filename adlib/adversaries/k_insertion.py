# k_insertion.py
# An implementation of the k-insertion attack where the attacker adds k data
# points to the model
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
import math
import numpy as np
from typing import List, Dict


# TODO: Implement gradient functions
# TODO: Implement gradient descent for 1 added vector
# TODO: Implement loop for k vectors using gradient descent


class KInsertion(Adversary):
    def __init__(self, learner, poison_instance, alpha, beta):
        Adversary.__init__(self)
        self.learner = learner
        self.poison_instance = poison_instance
        self.alpha = alpha
        self.beta = beta

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0:
            raise ValueError('Need at least one instance.')
        # TODO: Implement

    def _kernel_linear(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """
        Returns the value of the specified kernel function
        :param fv_1: feature vector 1 (np.ndarray)
        :param fv_2: feature vector 2 (np.ndarray)
        :return: the value of the specified kernel function
        """

        if len(fv_1) != len(fv_2):
            raise ValueError('Feature vectors need to have same length.')

        return fv_1.dot(fv_2)

    def _kernel_derivative_linear(self, fv_1: np.ndarray,
                                  fv_2: np.ndarray, k: int):
        """
        Returns the value of the derivative of the specified kernel function
        with fv_2 being the variable (i.e. K(x_i, x_c), finding gradient
        evaluated at x_c
        :param fv_1: fv_1: feature vector 1 (np.ndarray)
        :param fv_2: fv_2: feature vector 2 (np.ndarray)
        :param k: which partial derivative (0-based indexing, int)
        :return: the value of the derivative of the specified kernel function
        """

        if len(fv_1) != len(fv_2) or k < 0 or k >= len(fv_1):
            raise ValueError('Feature vectors need to have same '
                             'length and k must be a valid index.')

        return fv_1[k]

    def _kernel_poly(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """
        Returns the value of the specified kernel function
        :param fv_1: feature vector 1 (np.ndarray)
        :param fv_2: feature vector 2 (np.ndarray)
        :return: the value of the specified kernel function
        """

        if len(fv_1) != len(fv_2):
            raise ValueError('Feature vectors need to have same length.')

        return ((self.learner.gamma * fv_1.dot(fv_2) + self.learner.coef0) **
                self.learner.degree)

    def _kernel_derivative_poly(self, fv_1: np.ndarray,
                                fv_2: np.ndarray, k: int):
        """
        Returns the value of the derivative of the specified kernel function
        with fv_2 being the variable (i.e. K(x_i, x_c), finding gradient
        evaluated at x_c
        :param fv_1: fv_1: feature vector 1 (np.ndarray)
        :param fv_2: fv_2: feature vector 2 (np.ndarray)
        :param k: which partial derivative (0-based indexing, int)
        :return: the value of the derivative of the specified kernel function
        """

        if len(fv_1) != len(fv_2) or k < 0 or k >= len(fv_1):
            raise ValueError('Feature vectors need to have same '
                             'length and k must be a valid index.')

        return (fv_1[k] * self.learner.degree *
                self.learner.gamma *
                ((self.learner.gamma * fv_1.dot(fv_2) + self.learner.coef0) **
                 (self.learner.degree - 1)))

    def _kernel_rbf(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """
        Returns the value of the specified kernel function
        :param fv_1: feature vector 1 (np.ndarray)
        :param fv_2: feature vector 2 (np.ndarray)
        :return: the value of the specified kernel function
        """

        if len(fv_1) != len(fv_2):
            raise ValueError('Feature vectors need to have same length.')

        norm = np.linalg.norm(fv_1 - fv_2) ** 2
        return math.exp(-1 * self.learner.gamma * norm)

    def _kernel_derivative_rbf(self, fv_1: np.ndarray,
                               fv_2: np.ndarray, k: int):
        """
        Returns the value of the derivative of the specified kernel function
        with fv_2 being the variable (i.e. K(x_i, x_c), finding gradient
        evaluated at x_c
        :param fv_1: fv_1: feature vector 1 (np.ndarray)
        :param fv_2: fv_2: feature vector 2 (np.ndarray)
        :param k: which partial derivative (0-based indexing, int)
        :return: the value of the derivative of the specified kernel function
        """

        if len(fv_1) != len(fv_2) or k < 0 or k >= len(fv_1):
            raise ValueError('Feature vectors need to have same '
                             'length and k must be a valid index.')

        return (self._kernel_rbf(fv_1, fv_2) * 2 *
                self.learner.gamma * (fv_1[k] - fv_2[k]))

    def _kernel_sigmoid(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """
        Returns the value of the specified kernel function
        :param fv_1: feature vector 1 (np.ndarray)
        :param fv_2: feature vector 2 (np.ndarray)
        :return: the value of the specified kernel function
        """

        if len(fv_1) != len(fv_2):
            raise ValueError('Feature vectors need to have same length.')

        inside = self.learner.gamma * fv_1.dot(fv_2) + self.learner.coef0
        return math.tanh(inside)

    def _kernel_derivative_sigmoid(self, fv_1: np.ndarray,
                                   fv_2: np.ndarray, k: int):
        """
        Returns the value of the derivative of the specified kernel function
        with fv_2 being the variable (i.e. K(x_i, x_c), finding gradient
        evaluated at x_c
        :param fv_1: fv_1: feature vector 1 (np.ndarray)
        :param fv_2: fv_2: feature vector 2 (np.ndarray)
        :param k: which partial derivative (0-based indexing, int)
        :return: the value of the derivative of the specified kernel function
        """

        if len(fv_1) != len(fv_2) or k < 0 or k >= len(fv_1):
            raise ValueError('Feature vectors need to have same '
                             'length and k must be a valid index.')

        inside = self.learner.gamma * fv_1.dot(fv_2) + self.learner.coef0
        return self.learner.gamma * fv_1[k] / (math.cosh(inside) ** 2)

    def _get_kernel(self):
        """
        :return: the appropriate kernel function
        """

        if self.learner.kernel == 'linear':
            return self._kernel_linear
        elif self.learner.kernel == 'poly':
            return self._kernel_poly
        elif self.learner.kernel == 'rbf':
            return self._kernel_rbf
        elif self.learner.kernel == 'sigmoid':
            return self._kernel_sigmoid
        else:
            raise ValueError('No matching kernel function found.')

    def _get_kernel_derivative(self):
        """
        :return: the appropriate kernel derivative function
        """
        
        if self.learner.kernel == 'linear':
            return self._kernel_derivative_linear
        elif self.learner.kernel == 'poly':
            return self._kernel_derivative_poly
        elif self.learner.kernel == 'rbf':
            return self._kernel_derivative_rbf
        elif self.learner.kernel == 'sigmoid':
            return self._kernel_derivative_sigmoid
        else:
            raise ValueError('No matching kernel function found.')

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
