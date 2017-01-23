from adversaries.adversary import AdversaryStrategy
from typing import List, Dict
from data_reader.input import Instance, FeatureVector
import numpy as np
from learners.learner import InitialPredictor
from math import log

class Adversary(AdversaryStrategy):

    def __init__(self):
        self.xj_min = 0               # type: float (depends on feature domain size)
        self.xj_max = 1               # type: float (depends on feature domain size)
        self.f_attack = 1             # type: float (value between 0 and 1)
        self.innocuous_target         # need to figure out how to get a non malicious instance
        self.learn_model = None       # type: Classifier
        self.num_features = None      # type: int
        self.f_domain = 1             # 1 for binary features, 0 for non-binary features


    def change_instances(self, instances: List[Instance]) -> List[Instance]:
        transformed_instances = []
        if self.f_attack == 0:
            return instances
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == 1:
                transformed_instances.append(self.attack(transformed_instance))
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
        if 'f_domain' in params.keys():
            self.f_domain = params['f_domain']



    def get_available_params(self) -> Dict:
        params = {'xj_min': self.lambda_val,
                  'xj_max': self.max_change,
                  'f_attack': self.f_attack,
                  'f_domain': self.f_domain}
        return params

    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_vector().get_feature_count()
        self.innocuous_target = next(
            (x for x in instances if x.get_label() == InitialPredictor.negative_classification),
            None
        )
    # Maybe for the binary case, the f_attack value represents the percentage of features we change?
    def attack(self, instance: Instance):
        for i in range(0, self.num_features):
            delta_ij = self.innocuous_target - instance.get_feature_vector.get_feature(i)
            if delta_ij!=0:
                if self.f_domain == 1: # when features are binary
                    instance.get_feature_vector().flip_bit(i)
                else: # when we have non-binary features
                    instance.set_feature_weight(i,self.f_attack*delta_ij)
        return instance
