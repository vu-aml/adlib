from learners.learner import RobustLearner
from typing import Dict, List
from data_reader.dataset import EmailDataset
from learners.models.sklearner import Model
import numpy as np

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


class Retraining(RobustLearner):
    def __init__(self, base_model=None, training_instances:EmailDataset=None, params: Dict=None):
        RobustLearner.__init__(self)
        self.model = Model(base_model)
        self.attack_alg = None # Type: class
        self.adv_params = None
        self.attacker = None # Type: Adversary
        self.set_training_instances(training_instances)
        self.set_params(params)

    def set_params(self, params: Dict):
        if params['attack_alg'] is not None:
            self.attack_alg = params['attack_alg']
        if params['adv_params'] is not None:
            self.adv_params = params['adv_params']

    def train(self):
        self.model.train(self.training_instances)
        self.attacker = self.attack_alg()
        self.attacker.set_params(self.adv_params)
        self.attacker.set_adversarial_params(self.model,self.training_instances)
        print("training")
        malicious_feature_vecs = [x for x in self.training_instances.features if self.model.predict([x])[0] == 1]
        augmented_feature_vecs = self.training_instances.features
        augmented_labels = self.training_instances.labels

        for instance in malicious_feature_vecs:
            transformed_instance = self.attacker.attack([instance])[0]
            new_instance = True
            for old_instance in augmented_feature_vecs:
                if np.array_equal(old_instance, transformed_instance):
                    new_instance = False
            if new_instance:
                np.append(augmented_feature_vecs, transformed_instance, axis=0)
                np.append(augmented_labels, [1])
        self.model.train(EmailDataset(raw=False, features=augmented_feature_vecs, labels=augmented_labels))


    def decision_function(self, instances):
        return self.model.decision_function_adversary(instances)

    def predict(self, instances):
        """

        :param instances: matrix of instances shape (num_instances, num_feautres_per_instance)
        :return: list of labels (int)
        """
        return self.model.predict(instances)

    def predict_proba(self, instances):
        return self.model.predict_proba_adversary(instances)

