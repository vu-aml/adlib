# k_insertion.py
# An implementation of the k-insertion attack where the attacker adds k data
# points to the model
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from data_reader.binary_input import BinaryFeatureVector
import math
import pathos.multiprocessing as mp
import numpy as np
from copy import deepcopy
from typing import List, Dict


# TODO: Implement gradient functions
# TODO: Implement gradient descent for 1 added vector
# TODO: Implement loop for k vectors using gradient descent
# TODO: Save kernel function to self


class KInsertion(Adversary):
    def __init__(self, learner, poison_instance, beta=0.07, alpha=1e-3,
                 number_to_add=1):
        Adversary.__init__(self)
        self.learner = deepcopy(learner)
        self.poison_instance = poison_instance
        self.beta = beta
        self.alpha = alpha
        self.number_to_add = number_to_add
        self.instances = None
        self.orig_instances = None
        self.x = None  # The feature vector of the instance to be added
        self.y = -1 if np.random.binomial(1, 0.5, 1)[0] == 0 else 1  # x's label
        self.inst = None
        self.kernel = self._get_kernel()
        self.kernel_derivative = self._get_kernel_derivative()

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0:
            raise ValueError('Need at least one instance.')
        self.instances = deepcopy(instances)
        self.orig_instances = deepcopy(instances)

        # x is value to be added
        self.x = np.random.binomial(1, 0.5, instances[0].get_feature_count())
        indices_list = []
        for i in range(len(self.x)):
            if self.x[i] == 1:
                indices_list.append(i)
        self.inst = Instance(self.y,
                             BinaryFeatureVector(len(self.x), indices_list))

        # Train with newly generated instance
        self.instances.append(self.inst)
        self.learner.training_instances = self.instances
        self.learner.train()

        print(self.learner.model.learner.support_)

        gradient = []
        for i in range(instances[0].get_feature_count()):
            current = 0  # current partial derivative

            (z_c,
             partial_b_partial_x_k,
             partial_z_s_partial_x_k) = self._solve_parial_derivatives_matrix(i)

            print('z_c: ', z_c)
            print('pbpx_k: ', partial_b_partial_x_k)
            print('z_s partials: ', partial_z_s_partial_x_k)

            for j in range(len(self.orig_instances)):
                if j in self.learner.model.learner.support_:
                    q_i_t = self._Q(self.orig_instances[i], self.inst)
                    partial_z_i_partial_x_k = partial_z_s_partial_x_k[
                        self.learner.model.learner.support_.tolist().index(i)]
                    current += q_i_t * partial_z_i_partial_x_k

            current += self._Q(self.instances[-1], self.inst, True, i) * z_c

            if len(self.instances) in self.learner.model.learner.support_:
                current += (self._Q(self.instances[-1], self.inst) *
                            partial_z_s_partial_x_k[-1])

            current += self.inst.get_label() * partial_b_partial_x_k
            gradient.append(current)

        gradient = np.array(gradient)

        print(gradient)

    def _solve_parial_derivatives_matrix(self, k: int):
        """
        :return: z_c, partial b / partial x_k, partial z_s / partial x_k (tuple)
        """

        learner = self.learner.model.learner
        size = learner.n_support_[0] + learner.n_support_[1] + 1  # binary
        solution = np.full((size, size), 0)

        if len(self.instances) - 1 not in learner.support_:  # not in S
            if self.learner.predict(self.inst) != self.inst.get_label():  # in E
                z_c = learner.C
            else:  # in R, z_c = 0, everything is 0
                return 0, 0, np.full(learner.n_support_, 0)
        else:
            z_c = learner.coef_.flatten()[-1]

        y_s = []
        for i in learner.support_:
            y_s.append(self.instances[i].get_label())
        y_s = np.array(y_s)

        q_s = []
        pool = mp.Pool(mp.cpu_count())
        print('size - 1: ', size - 1)
        for i in range(size - 1):
            print('i: ', i)
            calculations = list(range(size - 1))
            pool.map(lambda index:
                     self._Q(self.instances[learner.support_[i]],
                             self.instances[learner.support_[index]]),
                     calculations)
            q_s.append(calculations)
        q_s = np.array(q_s)

        for i in range(1, size):
            solution[0][i] = y_s[i - 1]
            solution[i][0] = y_s[i - 1]

        for i in range(1, size):
            for j in range(1, size):
                solution[i][j] = q_s[i - 1][j - 1]

        solution = np.linalg.inv(solution)
        solution = -1 * z_c * solution

        vector = [0]
        for i in learner.support_:
            vector.append(self._Q(self.instances[learner.support_[i]],
                                  self.instances[-1], True, k))
        vector = np.array(vector)

        solution = solution * vector
        return z_c, solution[0], solution[1:]

    def _Q(self, inst_1: Instance, inst_2: Instance, derivative=False, k=-1):
        """
        Calculates Q_ij or partial Q_ij / partial x_k
        :param inst_1: the first instance
        :param inst_2: the second instance
        :param derivative: True -> calculate derivative, False -> calculate Q
        :param k: determines which derivative to calculate
        :return: Q_ij or the derivative where i corresponds to inst_1 and j
                 corresponds to inst_2
        """

        if inst_1.get_feature_count() != inst_2.get_feature_count():
            raise ValueError('Feature vectors need to have same length.')

        fv = [[], []]
        for i in range(2):
            if i == 0:
                inst = inst_1
            else:
                inst = inst_2

            for j in range(inst.get_feature_count()):
                if inst.get_feature_vector().get_feature(j) == 0:
                    fv[i].append(0)
                else:
                    fv[i].append(1)

        if derivative:
            ret_val = self.kernel_derivative(np.array(fv[0]),
                                             np.array(fv[1]),
                                             k)
        else:
            ret_val = self.kernel(np.array(fv[0]), np.array(fv[1]))
        return inst_1.get_label() * inst_2.get_label() * ret_val

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

        if self.learner.model.learner.kernel == 'linear':
            return self._kernel_linear
        elif self.learner.model.learner.kernel == 'poly':
            return self._kernel_poly
        elif self.learner.model.learner.kernel == 'rbf':
            return self._kernel_rbf
        elif self.learner.model.learner.kernel == 'sigmoid':
            return self._kernel_sigmoid
        else:
            raise ValueError('No matching kernel function found.')

    def _get_kernel_derivative(self):
        """
        :return: the appropriate kernel derivative function
        """

        if self.learner.model.learner.kernel == 'linear':
            return self._kernel_derivative_linear
        elif self.learner.model.learner.kernel == 'poly':
            return self._kernel_derivative_poly
        elif self.learner.model.learner.kernel == 'rbf':
            return self._kernel_derivative_rbf
        elif self.learner.model.learner.kernel == 'sigmoid':
            return self._kernel_derivative_sigmoid
        else:
            raise ValueError('No matching kernel function found.')

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
