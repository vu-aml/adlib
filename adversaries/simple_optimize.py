from adversaries.adversary import Adversary
# from data_reader.input import Instance, FeatureVector
from typing import List, Dict
from data_reader.dataset import EmailDataset
# from learners.models.model import BaseModel
# from classifier import Classifier
from copy import deepcopy
from math import exp

'''Simple optimization of adversarial instance with binary valued feature vector.

Concept:
    Iterates through features in the adversarial instance, flipping features
    that lower the probability of being classified adversarial.
'''


class SimpleOptimize(Adversary):

    def __init__(self, lambda_val=-100, max_change=1000, learner=None):
        Adversary.__init__(self)
        self.lambda_val = lambda_val                     # type: float
        self.max_change = max_change                     # type: float
        self.num_features = None                         # type: int
        self.learn_model = learner

    def attack(self, data: EmailDataset) -> EmailDataset:
        # transformed_instances = []
        if self.num_features is None:
            # self.num_features = instances[0].get_feature_vector().get_feature_count()
            self.num_features = len(data)
        for idx, label in enumerate(data.labels):
            # transformed_instance = deepcopy(instance)
            if label == 1:
                data.features[idx] = self.optimize(data.features.getrow(idx))
        return data

    def set_params(self, params: Dict):
        if 'lambda_val' in params.keys():
            self.lambda_val = params['lambda_val']
        if 'max_change' in params.keys():
            self.max_change = params['max_change']

    def get_available_params(self) -> Dict:
        params = {'lambda_val': self.lambda_val,
                  'max_change': self.max_change}
        return params

    def set_adversarial_params(self, learner, train_data: EmailDataset):
        self.learn_model = learner
        self.num_features = len(train_data)

    def optimize(self, instance):
        """Flip features that lower the prob. of being classified adversarial.
        Args:
            instance: (scipy.sparse.csr_matrix) feature vector

        """
        change = 0
        for i in range(0, self.num_features):
            orig_prob = self.learn_model.predict_proba(instance)[0]
            instance[0, i] = 0 if instance[0, i] else 1
            change += 1
            new_prob = self.learn_model.predict_proba(instance)[0]
            if new_prob >= (orig_prob-exp(self.lambda_val)):
                # flip the bit
                # TODO: flipbit method in EmailDataset
                instance[0, i] = 0 if instance[0, i] else 1
                change -= 1
            if change > self.max_change:
                break
        return instance
