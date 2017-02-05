from learners.learner import InitialPredictor, ImprovedPredictor
from typing import List
import numpy as np
from adversaries.feature_deletion import Adversary
from data_reader.input import Instance


class Learner(InitialPredictor):

    def __init__(self):
        InitialPredictor.__init__(self)


class ImprovedLearner(ImprovedPredictor):

    def __init__(self):
        ImprovedPredictor.__init__(self)
        self.weight_vector = None

    def predict(self, instances: List[Instance]):
        predictions = []
        for instance in instances:
            features = instance.get_feature_vector().get_csr_matrix().toarray()[0]
            predictions.append(np.sign(self.predict_instance(features)))
        return predictions

    def predict_instance(self, features):
        return np.dot(features, self.weight_vector)

    def set_adversarial_params(self, learner: Learner, adversary: Adversary):
        self.weight_vector = adversary.get_learner_params()
