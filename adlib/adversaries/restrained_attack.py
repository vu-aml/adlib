from adlib.adversaries.adversary import Adversary
from typing import List, Dict
from data_reader.binary_input import Instance
from data_reader.operations import find_centroid, find_max, find_min
from copy import deepcopy
import random

"""
Based on the Adversarial Support Vector Machine Learning by Yan Zhou, Murat Kantarcioglu,
 Bhavani Thuraisingham and Bowei Xi.

Concept: A generalized attacker algorithm that attempts to move the instances' features in a 
         certain direction by a certain distance that is measured by how harsh the attack is.
         This is more restrained than the free range attack because the distance is limited by
         some weight.
"""


class Restrained(Adversary):
    def __init__(self, f_attack=0.5, binary=True, discount_factor=1, type='random', learner=None):
        """

        :param f_attack:  float (between 0 and 1),determining the agressiveness
                          of the attack
        :param binary:    bool True means binary features
        :param learner:   from Learners
        :param type:      specify how to find innocuous target
        :param discount_factor: float(between 0 and 1),determing the data movement of the attack
        """
        self.f_attack = f_attack
        self.discount_factor = discount_factor
        self.innocuous_target = None
        self.num_features = None
        self.binary = binary
        self.learn_model = learner  # type: Classifier
        self.type = type

    def set_params(self, params: Dict):
        if 'f_attack' in params.keys():
            self.f_attack = params['f_attack']
        if 'binary' in params.keys():
            self.binary = params['binary']
        if 'learner' in params.keys():
            self.learn_model = params['learner']
        if 'type' in params.keys():
            self.type = params['type']

    def get_available_params(self) -> Dict:
        params = {'f_attack': self.f_attack,
                  'binary': self.binary,
                  'learner': self.learn_model,
                  'type': self.type}
        return params

    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_count()
        self.set_innocuous_target(train_instances, learner, self.type)

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
        for the real_value case, we generate a value between 0 and the bound.
        The bound is calculated by 1- c_delta * (abs(xt - x)/abs(x) + abs(xt)) * (xt -x)
        This value will be added to the xij for the new instance
        :param instance:
        :return: instance
        '''
        for i in range(0, self.num_features):
            xij = instance.get_feature_vector().get_feature(i)
            target = self.innocuous_target.get_feature_vector().get_feature(i)
            if abs(xij) + abs(target) == 0:
                bound = 0
            else:
                bound = self.discount_factor * (1 - self.f_attack *
                                                (abs(target - xij) /
                                                 (abs(xij) + abs(target)))) \
                        * abs((target - xij))
            # is that ok to just assign a random number between the range???
            delta_ij = random.uniform(0, bound)
            instance.flip(i, xij + delta_ij)
        return instance
