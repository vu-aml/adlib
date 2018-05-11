from adlib.learners.learner import learner
from typing import Dict, List
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance
from adlib.learners.models.sklearner import Model
import numpy as np
from data_reader.operations import fv_equals

"""Learner retraining.

Concept:
    Given a model used to train in the initial stage and access to
    make calls to adversarial transformation methods, proceeds by
    classifying the the given set of instances. Allows the adversary
    to iteratively transform the initial set of bad instances. While the
    adversary is capable of changing a negative instance to a positive
    instance, retrains and notifies the adversary of the change.

    After the improvement finishes, the underlying learner model has
    been updated, and can be used in the default prediction method.

"""


class Retraining(learner):
    def __init__(self, base_model=None, training_instances=None, attacker = None,params: Dict = None):
        learner.__init__(self)
        self.model = Model(base_model)
        #self.attack_alg = None  # Type: class
        #self.adv_params = None
        self.attacker = attacker  # Type: Adversary
        self.set_training_instances(training_instances)
        self.iteration_times = 5  # int: control the number of rounds directly
        #self.set_params(params)

    def set_params(self, params: Dict):
        #if 'attack_alg' in params.keys():
        #    self.attack_alg = params['attack_alg']
        if 'attacker' in params.keys():
            self.attacker = params['attacker']
        #if params['adv_params'] is not None:
         #   self.adv_params = params['adv_params']
        if 'iteration_times' in params.keys() and 'iteration_times' is not None:
            self.iteration_times = params['iteration_times']

    def get_available_params(self) -> Dict:
        params = {'attacker':self.attacker,
                  'attack_params':self.adv_params,
                  'iteration_times': self.iteration_times}
        return params

    def train(self):
        '''
        This is implemented according to Algorithm 1 in Central Rettraining Framework
        for Scalable Adversarial Classification. This will iterate between computing
        a classifier and adding the adversarial instances to the training data that evade
        the previously computed classifier.
        :return: None
        '''
        self.model.train(self.training_instances)
        iteration = self.iteration_times
        #self.attacker = self.attack_alg()
        #self.attacker.set_params(self.adv_params)
        #self.attacker.set_adversarial_params(self.model, self.training_instances)

        print("==> Training...")
        malicious_instances = [x for x in self.training_instances if
                               self.model.predict(x) == 1]
        augmented_instances = self.training_instances

        while iteration != 0:
            new = []
            transformed_instances = self.attacker.attack(malicious_instances)
            for instance in transformed_instances:
                in_list = False
                for idx, old_instance in enumerate(augmented_instances):
                    if fv_equals(old_instance.get_feature_vector(),instance.get_feature_vector()):
                        in_list = True
                if not in_list:
                    new.append(instance)
                augmented_instances.append(Instance(label=1, feature_vector=instance.get_feature_vector()))
            self.model.train(augmented_instances)
            malicious_instances = [x for x in augmented_instances if
                                   self.model.predict(x) == 1]
            iteration -= 1
            if new is None:
                break

    def decision_function(self, instances):
        return self.model.decision_function_(instances)

    def predict(self, instances):
        """

        :param instances: matrix of instances shape (num_instances, num_feautres_per_instance)
        :return: list of labels (int)
        """
        return self.model.predict(instances)

    def predict_proba(self, instances):
        return self.model.predict_proba(instances)

    def get_weight(self):
        return self.model.learner.coef_[0]

    def get_constant(self):
        return self.model.learner.intercept_
