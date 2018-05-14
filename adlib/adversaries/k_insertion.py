# k_insertion.py
# An implementation of the k-insertion attack where the attacker adds k data
# points to the model
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
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
    
    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
