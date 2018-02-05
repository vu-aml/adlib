from adversaries.adversary import Adversary
from typing import List, Dict
from data_reader.binary_input import Instance
from copy import deepcopy
import numpy as np
import math

"""
Based on the Adversarial learning by Daniel Loaw and Christopher Meek.
Implementation of the adversarial classifier reverse engineering (ACRE) learning problem that
can solve input data of real value instances and binary instances.
"""


class AdversarialLearning(Adversary):
    def __init__(self, cost_weight=None, binary=True, threshold=0.5, learner=None):
        self.cost_weight = cost_weight
        self.threshold = threshold
        self.binary = binary
        self.learner = learner

    def set_params(self, params: Dict):
        if 'cost_weight' in params.keys():
            self.f_attack = params['cost_weight']
        if 'binary' in params.keys():
            self.binary = params['binary']
        if 'learner' in params.keys():
            self.learn_model = params['learner']
        if 'threshold' in params.keys():
            self.type = params['threshold']

    def get_available_params(self) -> Dict:
        params = {'cost_weight': self.f_attack,
                  'binary': self.binary,
                  'learner': self.learner,
                  'threshold': self.type}
        return params

    def set_adversarial_params(self, learner, training_instances: List[Instance]):
        self.learner = learner
        self.num_features = training_instances[0].get_feature_count()

    def attack(self, instances: List[Instance]) -> List[Instance]:
        transformed_instances = []
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == self.learner.positive_classification:
                transformed_instances.append(self.find_continuous_imac(xa= transformed_instance,step_size=10))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def find_continuous_imac(self, xa, step_size):
        """
        find the feature with largest weight/cost, and update the according feature
        with minimum changes to change its classification into negative
        :param xa:
        :param step_size:
        :return:
        """
        weight = self.learner.model.learner.coef_
        biggest_weight = 0
        biggest_index = 0
        for i in range(self.num_features):
            if self.cost_weight is None:
                current_weight = math.fabs(weight[i])
            else:
                current_weight = math.fabs(weight[i]) / self.cost_weight[i]
            if current_weight > biggest_weight:
                biggest_weight = current_weight
                biggest_index = i
        return self.line_search(xa =xa, weight=biggest_weight, i=biggest_index, step_size=step_size)

    def line_search(self, weight, xa, i, step_size):
        """
        :param xa: attack instance
        :param i: the feature with largest weight
        :param step_size:
        :return:
        """
        # find the maximum and minimum, and perform a binary search
        min = xa.get_feature_vector().get_feature(i)
        max = (weight * min + self.threshold / 4) / weight
        val_list = np.linspace(min, max, step_size)
        for val in val_list:
            x_prime = deepcopy(xa)
            x_prime.flip(i, min + val)
            if self.learner.predict(x_prime) == self.learner.negative_classification:
                return x_prime
        return xa

    def gap(self, x: Instance, weight):
        """
        :param x: Instance object
        :param weight: a vector specifying linear weights
        :return: real value of gap(x)
        """
        x_prime = x.get_csr_matrix().toarray().T
        return math.fabs(weight * x_prime - self.threshold)
