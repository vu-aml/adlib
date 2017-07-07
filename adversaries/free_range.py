from adversaries.adversary import Adversary
from typing import List, Dict
from data_reader.binary_input import Instance, FeatureVector
from copy import deepcopy
import numpy as np
from learners.learner import learner
from math import log

class FreeRange(Adversary):

    def __init__(self, xj_min = 0.0, xj_max = 1.0,f_attack = 0.5, binary = True, learner=None):
        self.xj_min = xj_min               # type: float (depends on feature domain size)
        self.xj_max = xj_max               # type: float (depends on feature domain size)
        self.f_attack = f_attack           # type: float (between 0 and 1),determining the agressiveness
                                           # of the attack
        self.innocuous_target = None       # need to figure out how to get a non malicious instance
        self.num_features = None           # type: int
        self.binary = binary               # type: bool True means binary features
        self.learn_model=learner	#type: Classifier


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

    def set_params(self, params: Dict):
        if 'xj_min' in params.keys():
            self.xj_min = params['xj_min']
        if 'xj_max' in params.keys():
            self.xj_max = params['xj_max']
        if 'f_attack' in params.keys():
            self.f_attack = params['f_attack']
        if 'binary' in params.keys():
            self.binary = params['binary']



    def get_available_params(self) -> Dict:
        params = {'xj_min': self.xj_min,
                  'xj_max': self.xj_max,
                  'f_attack': self.f_attack,
                  'binary': self.binary}
        return params


    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_vector().get_feature_count()
        #may take the centroid of innocuous data if the data is not binary
        self.innocuous_target = next(
            (x for x in train_instances if x.get_label() == learner.negative_classification),
            None
        )

    def transform(self, instance: Instance):
        # Maybe for the binary case, the f_attack value represents the percentage of features we change.
        # If f_attack =1, then the result should be exactly the same as innocuous target.
        attack_times = (int)(self.f_attack * self.num_features)
        count = 0
        for i in range(0, self.num_features):
            delta_ij = self.innocuous_target.get_feature_vector().get_feature(i) \
                       - instance.get_feature_vector().get_feature(i)
            if delta_ij!=0:
                if self.binary: # when features are binary
                    instance.get_feature_vector().flip_bit(i)
                else: # when we have non-binary features
                    instance.set_feature_weight(i,self.f_attack*delta_ij)
            count += 1
            if count == attack_times:
                return instance
        return instance
