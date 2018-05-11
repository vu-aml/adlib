from adlib.adversaries.adversary import Adversary
from typing import List, Dict
from data_reader.binary_input import Instance
from data_reader.operations import find_centroid, find_max, find_min
from copy import deepcopy
import numpy as np
from adlib.learners.learner import learner
from math import log
import random

"""
Based on the Adversarial Support Vector Machine Learning by Yan Zhou, Murat Kantarcioglu,
 Bhavani Thuraisingham and Bowei Xi.
 
Concept: A generalized attacker algorithm that attempts to move the instances' features in a 
         certain direction by a certain distance that is measured by how harsh the attack is.
                                                          
"""


class FreeRange(Adversary):
    def __init__(self, f_attack=0.5, xj_min=0.0, xj_max=0.0, type = 'random',binary=False, learner=None):
        """

        :param f_attack:  float (between 0 and 1),determining the agressiveness
                          of the attack
        :param xj_min:    minimum xj that the feature can have
                          If not specified, it is calculated by going over all training data.
        :param xj_max:    maximum xj that the feature can have
                          If not specified, it is calculated by going over all training data.
        :param binary:    bool True means binary features
        :param learner:   from Learners
        :param type:      specify how to find innocuous target
        """
        self.xj_min = xj_min
        self.xj_max = xj_max
        self.f_attack = f_attack
        self.innocuous_target = None
        self.num_features = None
        self.binary = binary
        self.type = type
        self.learn_model = learner  # type: Classifier


    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_count()
        self.set_innocuous_target(train_instances, learner, self.type)
        self.set_boundaries(train_instances)

    def set_params(self, params: Dict):
        if 'xj_min' in params.keys():
            self.xj_min = params['xj_min']
        if 'xj_max' in params.keys():
            self.xj_max = params['xj_max']
        if 'f_attack' in params.keys():
            self.f_attack = params['f_attack']
        if 'binary' in params.keys():
            self.binary = params['binary']
        if 'type' in params.keys():
            self.type = params['type']

    def get_available_params(self) -> Dict:
        params = {'xj_min': self.xj_min,
                  'xj_max': self.xj_max,
                  'f_attack': self.f_attack,
                  'binary': self.binary,
                  'type': self.type
                  }
        return params


    def attack(self, instances: List[Instance]) -> List[Instance]:
        transformed_instances = []
        if self.f_attack == 0:
            return instances
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == 1:
                transformed_instances.append(self.transform(transformed_instance))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def set_boundaries(self, train_instances):
        """
        Setting the x_min and x_max by estimating the smallest and largest
        value from the training data.
        :param train_instances:
        :return: None
        """
        self.x_min = find_min(train_instances)
        self.x_max = find_max(train_instances)

    def set_innocuous_target(self, train_instances, learner, type):
        """
        If type is random, we simply pick the first instance from training data as the
        innocuous target. Otherwise, we compute the centroid of training data.
        :param train_instances:
        :param learner:
        :param type: specifies how to find innocuous_target
        :return: None
        """
        if type == 'random':
            self.innocuous_target = next(
                (x for x in train_instances if x.get_label() == learner.negative_classification),
                None)
        elif type == 'centroid':
            target = find_centroid(train_instances)
            if learner.predict(target) == 1:
                print("Fail to find centroid of from estimated training data")
                self.innocuous_target = next(
                    (x for x in train_instances if x.get_label() == learner.negative_classification),
                    None)
            else:
                self.innocuous_target = target



    def transform(self, instance: Instance):
        '''
        for the binary case, the f_attack value represents the percentage of features we change.
        If f_attack =1, then the result should be exactly the same as innocuous target.

        for the real_value case, we generate a value between c_f(x_min - xij) and c_f(x_max - xij)
        This value will be added to the xij for the new instance
        :param instance:
        :return: instance
        '''
        if self.binary:
            attack_times = (int)(self.f_attack * self.num_features)
            count = 0
            for i in range(0, self.num_features):
                delta_ij = self.innocuous_target.get_feature_vector().get_feature(i) \
                           - instance.get_feature_vector().get_feature(i)
                if delta_ij != 0:
                    if self.binary:  # when features are binary
                        instance.get_feature_vector().flip_bit(i)
                count += 1
                if count == attack_times:
                    return instance
        else:
            for i in range(0, self.num_features):
                xij = instance.get_feature_vector().get_feature(i)
                if self.xj_min == 0 and self.xj_max == 0:
                    lower_bound = self.f_attack * (self.x_min.get_feature(i) - xij)
                    upper_bound = self.f_attack * (self.x_max.get_feature(i) - xij)
                else:
                    lower_bound = self.f_attack * (self.xj_min - xij)
                    upper_bound = self.f_attack * (self.xj_max - xij)
                # is that ok to just assign a random number between the range???
                delta_ij = random.uniform(lower_bound, upper_bound)
                instance.flip(i, xij + delta_ij)
        return instance
