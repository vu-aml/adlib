from adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from typing import List, Dict
import numpy as np
from copy import deepcopy
from learners import SimpleLearner
from sklearn.svm import SVC
from sklearn.metrics import pairwise
#import matplotlib.pyplot as plt

"""
  Based on Nightmare at Test Time: Robust Learning by Feature Deletion by Amir Globerson
  and Sam Roweis.
  
  Concept: Implementing a typical attacker that tries to delete the features with the least
           weights by setting the features' value to zero to fool the learning algorithm.
"""

class AdversaryFeatureDeletion(Adversary):
    def __init__(self, learner=None, num_deletion=100, all_malicious=False,random = False):
        """
        :param learner: Learner from learners
        :param num_deletion: the max number that will be deleted in the attack
        :param all_malicious: if the flag is set, only features that are malicious
                              will be deleted.
        """
        Adversary.__init__(self)
        self.num_features = 0  # type: int
        self.num_deletion = num_deletion  # type: int
        self.malicious = all_malicious  # type: bool
        self.learn_model = learner
        self.del_index = None  # type: np.array
        self.random = random
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

        rbf_flag = self.is_rbf()

        if self.weight_vector is None and not rbf_flag:
            print("Can not acquire weight vector from the learning model")
            print("Set random = True: randomly delete features instead")
            self.random = True

        attacked_instances = []

        #for rbf kernel,we can delete the features with the most gradient
        if rbf_flag and not self.random:
            for instance in instances:
                if instance.label < 0:
                    attacked_instances.append(instance)
                else:
                    gradient = self.rbf_kernel_gradient(instance)
                    if self.malicious:
                        self.del_index = np.flipud(np.argsort(gradient))[:self.num_deletion]
                    else:
                        self.del_index = np.flipud(np.argsort(np.absolute(gradient)))[:self.num_deletion]
                    attacked_instances.append(self.change_instance(instance))
            return attacked_instances
        else:
            if not self.random:
                if self.malicious:
                    self.del_index = np.flipud(np.argsort(self.weight_vector))[:self.num_deletion]
                else:
                    self.del_index = np.flipud(np.argsort(np.absolute(self.weight_vector)))[:self.num_deletion]
            else:
                random_seed = np.random.random_sample((self.num_features,))
                self.del_index = np.argsort(random_seed)[:self.num_deletion]

            for instance in instances:
                if instance.label < 0:
                    attacked_instances.append(instance)
                else:
                    attacked_instances.append(self.change_instance(instance))
            return attacked_instances



    def set_params(self, params: Dict):
        if 'num_deletion' in params:
            self.num_deletion = params['num_deletion']
        if 'all_malicious' in params:
            self.malicious = params['all_malicious']
        if 'random' in params:
            self.random = params['random']

    def get_available_params(self) -> Dict:
        params = {'num_deletion': self.num_deletion,
                  'all_malicious': self.malicious,
                  'random':self.random}
        return params

    def change_instance(self, instance: Instance) -> Instance:
        instance_prime = deepcopy(instance)
        #x = instance.get_feature_vector().get_csr_matrix().toarray()[0]
        for i in range(0, self.num_features):
            if i in self.del_index:
                instance_prime.flip(i, 0)
        return instance_prime

        # the return value should not be 1 here, since benign instances can change as well?
        # indices = [i for i in range(0, self.num_features) if x[i] != 0 and i not in self.del_index]
        # return Instance(1, FeatureVector(self.num_features, indices))

    #for RBF kernel
    #can not obtain weight here, so we can just calculate the gradient for the rbf kernel
    #based on the gradient, delete the features with the most weight

    def is_rbf(self):
        if type(self.learn_model) != SimpleLearner or type(self.learn_model.model.learner) != SVC\
                or type(self.learn_model.model.learner.kernel != "rbf"):
            return False
        return True

    def rbf_kernel_gradient(self,attack_instance):
        param_map = self.learn_model.get_params()
        attribute_map = self.learn_model.get_attributes()
        if param_map["kernel"] == "rbf":
            grad = []
            dual_coef = attribute_map["dual_coef_"]
            support = attribute_map["support_vectors_"]
            gamma = param_map["gamma"]
            kernel = pairwise.rbf_kernel(support, attack_instance, gamma)
            for element in range(0, len(support)):
                if grad == []:
                    grad = (dual_coef[0][element] * kernel[0][element] * 2 * gamma * (support[element] -
                                                                                      attack_instance))
                else:
                    grad = grad + (
                        dual_coef[0][element] * kernel[element][0] * 2 * gamma * (support[element] -
                                                                                  attack_instance))
            return np.array(-grad)

