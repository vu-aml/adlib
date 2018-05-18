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


class DataModification(Adversary):
    def __init__(self, learner, verbose=False):
        Adversary.__init__(self)
        self.learner = deepcopy(learner)
        self.verbose = verbose
        self.fvs = None  # feature vector matrix, shape: (# inst., # features)
        self.theta = None
        self.b = None
        self.g_arr = None  # array of g_i values, shape: (# inst.)

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0:
            raise ValueError('Need at least one instance.')

        raise NotImplementedError

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
        self.theta = []
        for i in range(len(instances)):
            if i in learner.support_:  # in S
                index = learner.support_.tolist().index(i)
                self.theta.append(learner.dual_coef_.flatten()[index])
            else:  # not in S
                if (self.learner.predict([instances[i]])[0] !=
                        instances[i].get_label()):  # in E
                    self.theta.append(learner.C)
                else:  # in R
                    self.theta.append(0)
        self.theta = np.array(self.theta)

        self.b = learner.intercept_[0]
        self.g_arr = learner.decision_function(self.fvs)

    @staticmethod
    def _logistic_function(x):
        return 1 / (1 + math.exp(-1 * x))

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
