from adlib.adversaries.adversary import Adversary
from typing import List, Dict
from data_reader.binary_input import Instance
from adlib.learners.learner import Learner
from copy import deepcopy
from math import exp

"""Simple optimization of adversarial instance with binary valued feature 
vector.

Concept:
    Iterates through features in the adversarial instance, flipping features
    that lower the probability of being classified adversarial.
"""


class SimpleOptimize(Adversary):
    def __init__(self, lambda_val=-100, max_change=1000, learner=None):
        Adversary.__init__(self)
        self.lambda_val = lambda_val  # type: float
        self.max_change = max_change  # type: float
        self.num_features = None  # type: int
        self.learn_model = learner

    def attack(self, instances: List[Instance]) -> List[Instance]:
        transformed_instances = []
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == Learner.positive_classification:
                transformed_instances.append(
                    self.optimize(transformed_instance))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def set_params(self, params: Dict):
        if 'lambda_val' in params.keys():
            self.lambda_val = params['lambda_val']
        if 'max_change' in params.keys():
            self.max_change = params['max_change']

    def get_available_params(self) -> Dict:
        params = {'lambda_val': self.lambda_val,
                  'max_change': self.max_change}
        return params

    def set_adversarial_params(self, learner, training_data):
        self.learn_model = learner
        self.num_features = training_data[
            0].get_feature_vector().get_feature_count()

    def optimize(self, instance: Instance):
        """Flip features that lower the prob. of being classified adversarial.
        Args:
            instance: (scipy.sparse.csr_matrix) feature vector

        """
        change = 0
        for i in range(0, self.num_features):
            orig_prob = self.learn_model.predict_proba([instance])[0]
            new_instance = deepcopy(instance)
            new_instance.get_feature_vector().flip_bit(i)
            new_prob = self.learn_model.predict_proba([new_instance])[0]
            if new_prob < (orig_prob - exp(self.lambda_val)):
                instance.get_feature_vector().flip_bit(i)
                change += 1
            if change > self.max_change:
                break
        return instance
