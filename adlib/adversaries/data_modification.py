# data_modification.py
# An implementation of the data modification attack where the attacker modifies
# certain feature vectors
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from copy import deepcopy
from typing import List, Dict
import math
import numpy as np
import pathos.multiprocessing as mp


# TODO: self._calc_theta(self) is NOT done
# TODO: Use numpy.eye in calculate theta
# TODO: Search for other TODO


class DataModification(Adversary):
    def __init__(self, learner, target_theta, alpha=1e-3, beta=0.1,
                 verbose=False):

        Adversary.__init__(self)
        self.learner = deepcopy(learner).model.learner
        self.target_theta = target_theta
        self.alpha = alpha
        self.beta = beta  # learning rate
        self.verbose = verbose
        self.instances = None
        self.orig_fvs = None  # same as below, just original values
        self.fvs = None  # feature vector matrix, shape: (# inst., # features)
        self.theta = None
        self.b = None
        self.g_arr = None  # array of g_i values, shape: (# inst.)
        self.labels = None  # array of labels of the instances
        self.logistic_vals = None

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0:
            raise ValueError('Need at least one instance.')

        self.instances = instances
        self._calculate_constants(instances)

    def _calculate_constants(self, instances: List[Instance]):
        # Calculate feature vectors as np.ndarrays
        self.fvs = []
        for i in range(len(instances)):
            feature_vector = instances[i].get_feature_vector()
            tmp = []
            for j in range(instances[0].get_feature_count()):
                if feature_vector.get_feature(j) == 1:
                    tmp.append(1)
                else:
                    tmp.append(0)
            tmp = np.array(tmp)
            self.fvs.append(tmp)
        self.fvs = np.array(self.fvs)
        self.orig_fvs = deepcopy(self.fvs)

        # Calculate labels
        self.labels = []
        for inst in instances:
            self.labels.append(inst.get_label())
        self.labels = np.array(self.labels)

        # Calculate theta, b, and g_arr
        self.g_arr = self.learner.decision_function(self.fvs)
        self.b = self.learner.intercept_[0]
        self._calc_theta()

        # Calculate logistic function values - sigma(y_i g_i)
        self.logistic_vals = [
            DataModification._logistic_function(self.labels[i] * self.g_arr[i])
            for i in range(len(self.instances))]
        self.logistic_vals = np.array(self.logistic_vals)

    def _calc_theta(self):
        self.learner.fit(self.fvs, self.labels)  # Retrain learner

        self.theta = []
        for i in range(self.instances[0].get_feature_count()):
            std_basis_vect = []
            for _ in range(i):
                std_basis_vect.append(0)
            std_basis_vect.append(1)
            for _ in range(self.instances[0].get_feature_count() - i - 1):
                std_basis_vect.append(0)
            std_basis_vect = np.array(std_basis_vect)
            self.theta.append(std_basis_vect)
        self.theta = np.array(self.theta)

        self.theta = self.learner.decision_function(self.theta)
        self.theta = self.theta - self.b

    def _calc_gradient(self):
        matrix_1 = self._calc_partial_f_partial_capital_d()
        matrix_2 = np.linalg.inv(self._calc_partial_f_partial_theta())
        partial_theta_partial_capital_d = -1 * matrix_1.dot(matrix_2)

        # Calculate first part
        risk_gradient = 2 * (self.theta - self.target_theta)
        gradient = risk_gradient.dot(partial_theta_partial_capital_d)
        gradient *= self.beta
        gradient = [gradient for _ in range(len(self.instances))]
        gradient = np.array(gradient)

        # Calculate cost part
        cost = np.linalg.norm(self.fvs - self.orig_fvs)
        if cost > 0:  # cost can never be < 0
            cost_gradient = self.fvs - self.orig_fvs
            cost_gradient /= cost
        else:
            cost_gradient = 0

        return np.array(gradient + cost_gradient)

    def _calc_partial_f_partial_theta(self):
        pool = mp.Pool(mp.cpu_count())
        matrix = pool.map(lambda j: list(map(
            lambda k: self._calc_partial_f_j_partial_theta_k(j, k),
            range(len(self.theta)))), range(len(self.theta)))

        return np.array(matrix)

    def _calc_partial_f_j_partial_theta_k(self, j, k):
        running_sum = 0
        for i in range(len(self.instances)):
            val = self.logistic_vals[i]
            running_sum += self.fvs[i][k] * self.fvs[i][j] * val * (1 - val)

        return running_sum

    def _calc_partial_f_partial_capital_d(self):
        pool = mp.Pool(mp.cpu_count())
        matrix = pool.map(lambda j: list(map(
            lambda k: self._calc_partial_f_j_partial_x_k(j, k),
            range(len(self.theta)))), range(len(self.theta)))

        return np.array(matrix)

    def _calc_partial_f_j_partial_x_k(self, j, k):
        # TODO: Possibly bad formula!!!!!!!!!!!!!!
        running_sum = 0
        for i in range(len(self.instances)):
            val = self.logistic_vals[i]
            running_sum += (val * (1 - val) * self.labels[i] * self.fvs[i][j] *
                            self.theta[k])
            running_sum -= 1
            if j == k:
                running_sum += val * self.labels[i]

        return running_sum

    @staticmethod
    def _logistic_function(x):
        return 1 / (1 + math.exp(-1 * x))

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
