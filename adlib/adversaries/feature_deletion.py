from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from typing import List, Dict
import numpy as np
from copy import deepcopy

"""
  Based on Nightmare at Test Time: Robust Learning by Feature Deletion by 
  Amir Globerson and Sam Roweis.
  
  Concept: Implementing a typical attacker that tries to delete the features 
           with the least weights by setting the features' value to zero to 
           fool the learning algorithm.
"""


class AdversaryFeatureDeletion(Adversary):
    def __init__(self, learner=None, num_deletion=100, all_malicious=True):
        """
        :param learner: Learner from adlib.learners
        :param num_deletion: the max number that will be deleted in the attack
        :param all_malicious: if the flag is set, only features that are
                              malicious will be deleted.
        """
        Adversary.__init__(self)
        self.num_features = 0  # type: int
        self.num_deletion = num_deletion  # type: int
        self.malicious = all_malicious  # type: bool
        self.learn_model = learner
        self.del_index = None  # type: np.array
        if self.learn_model is not None:
            self.weight_vector = self.learn_model.get_weight()
        else:
            self.weight_vector = None  # type: np.array

    def set_adversarial_params(self, learner, train_instances):
        self.learn_model = learner
        self.weight_vector = self.learn_model.get_weight()

    def attack(self, instances: List[Instance]) -> List[Instance]:
        if self.weight_vector is None and self.learn_model is not None:
            self.weight_vector = self.learn_model.get_weight()
        if self.num_features == 0:
            self.num_features = instances[0].get_feature_count()
        if self.weight_vector is None:
            raise ValueError(
                'Must set learner_model and weight_vector before attack.')
        if self.malicious:
            self.del_index = np.flipud(np.argsort(self.weight_vector))[
                             :self.num_deletion]
        else:
            self.del_index = np.flipud(
                np.argsort(np.absolute(self.weight_vector)))[:self.num_deletion]

        # print("checking feature deletion attacker:")
        # print(self.del_index)
        # for i in self.del_index:
        #     print("the number {0} feature weight is {1}".format(i,
        #           self.weight_vector[i]))
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
        instance_prime = deepcopy(instance)
        # x = instance.get_feature_vector().get_csr_matrix().toarray()[0]
        for i in range(0, self.num_features):
            if i in self.del_index:
                instance_prime.flip(i, 0)
        return instance_prime

        # the return value should not be 1 here, since benign instances can
        # change as well?
        # indices = [i for i in range(0, self.num_features) if x[i] != 0 and
        # i not in self.del_index]
        # return Instance(1, FeatureVector(self.num_features, indices))
