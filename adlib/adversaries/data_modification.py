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


class DataModification(Adversary):
    def __init__(self, learner, verbose=False):
        Adversary.__init__(self)
        self.learner = deepcopy(learner)
        self.verbose = verbose
        self.instances = None
        self.fvs = None  # feature vector matrix, shape: (# inst., # features)
        self.theta = None
        self.b = None
        self.g_arr = None  # array of g_i values, shape: (# inst.)
        self.labels = None  # array of labels of the instances

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0:
            raise ValueError('Need at least one instance.')

        self.instances = instances
        self._calculate_constants(instances)
        matrix = self._calc_partial_f_partial_capital_d()
        print(matrix)

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

        # Calculate theta, b, and g_arr
        learner = self.learner.model.learner
        self.g_arr = learner.decision_function(self.fvs)
        self.b = learner.intercept_[0]

        self.theta = []
        for i in range(instances[0].get_feature_count()):
            std_basis_vect = []
            for _ in range(i):
                std_basis_vect.append(0)
            std_basis_vect.append(1)
            for _ in range(instances[0].get_feature_count() - i - 1):
                std_basis_vect.append(0)
            std_basis_vect = np.array(std_basis_vect)
            self.theta.append(std_basis_vect)
        self.theta = np.array(self.theta)

        self.theta = learner.decision_function(self.theta)
        self.theta = self.theta - self.b

        # Calculate labels
        self.labels = []
        for inst in instances:
            self.labels.append(inst.get_label())
        self.labels = np.array(self.labels)

    def _calc_partial_f_partial_capital_d(self):
        matrix = list(map(lambda j: list(
            map(lambda k: self._calc_partial_f_j_partial_x_k(j, k),
                range(len(self.instances)))),
                          range(len(self.instances))))
        return np.array(matrix)

    def _calc_partial_f_j_partial_x_k(self, j, k):
        # TODO: Possibly bad formula!!!!!!!!!!!!!!
        running_sum = 0
        for i in range(len(self.instances)):
            val = self._logistic_function(self.labels[i] * self.g_arr[i])
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
