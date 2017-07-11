from adversaries.adversary import Adversary
from data_reader.binary_input import Instance, FeatureVector
from typing import List, Dict
from random import shuffle
import numpy as np
from copy import deepcopy


class AdversaryFeatureDeletion(Adversary):
    def __init__(self, learner=None, num_deletion=0, all_malicious=False):
        Adversary.__init__(self)
        self.num_features = 0  # type: int
        self.num_deletion = num_deletion  # type: int
        self.malicious = all_malicious  # type: bool
        self.learn_model = learner
        self.del_index = None  # type: np.array
        if self.learn_model is not None:
            self.weight_vector = self.learn_model.model.learner.coef_.toarray()[0]
        else:
            self.weight_vector = None  # type: np.array

    def attack(self, instances: List[Instance]) -> List[Instance]:

        if self.weight_vector is None and self.learn_model is not None:
            self.weight_vector = self.learn_model.model.learner.coef_.toarray()[0]
        if self.num_features == 0:
            self.num_features = instances[0].get_feature_vector().get_feature_count()
        if self.weight_vector is None:
            raise ValueError('Must set learner_model and weight_vector before attack.')
        if self.malicious:
            # if malicious, only features that indicates malicious instance are deleted
            self.del_index = np.flipud(np.argsort(self.weight_vector))[:self.num_deletion]
        else:
            self.del_index = np.flipud(np.argsort(np.absolute(self.weight_vector)))[:self.num_deletion]

        return [self.change_instance(ins) for ins in instances]

    def set_params(self, params: Dict):
        if 'num_deletion' in params:
            self.num_deletion = params['num_deletion']

        if 'all_malicious' in params:
            self.malicious = params['all_malicious']

    def get_available_params(self) -> Dict:
        params = {'num_deletion': self.num_deletion,
                  'all_malicious': self.malicious, }
        return params

    def change_instance(self, instance: Instance) -> Instance:

        x = instance.get_feature_vector().get_csr_matrix().toarray()[0]
        indices = [i for i in range(0, self.num_features) if x[i] == 1 and i not in self.del_index]
        return Instance(1, FeatureVector(self.num_features, indices))
