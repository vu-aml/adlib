from learners.learner import RobustLearner
from typing import Dict, List
from data_reader.input import Instance
from learners.models.sklearner import Model
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


class Retraining(RobustLearner):
    def __init__(self, base_model=None, training_instances=None, params: Dict=None):
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
        I_bad = [x for x in self.training_instances if self.model.predict([x])[0] == 1]
        N = []
        while True:
            new = []
            for instance in I_bad:
                transformed_instance = self.attacker.attack([instance])[0]
                new_instance = True
                for old_instance in N:
                    if fv_equals(transformed_instance.get_feature_vector(),
                                 old_instance.get_feature_vector()):
                        new_instance = False
                if new_instance:
                    new.append(transformed_instance)
                    N.append(transformed_instance)
            if len(new) == 0:
                break
            self.model.train(self.training_instances + N)
            break

    def decision_function(self, instances):
        return self.model.decision_function_adversary(instances)

    def predict(self, instances: List[Instance]):
        return self.model.predict(instances)

    def predict_proba(self, instances: List[Instance]):
        return self.model.predict_proba_adversary(instances)

