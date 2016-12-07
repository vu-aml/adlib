from learners.learner import InitialPredictor, ImprovedPredictor
from adversaries.adversary import AdversaryStrategy
from typing import Dict, List
from types import FunctionType
import numpy as np
from data_reader.input import FeatureVector
from learners.models.model import BaseModel
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

class Learner(InitialPredictor):

    def __init__(self):
        InitialPredictor.__init__(self)

    def decision_function(self, instances):
        return self.get_model().decision_function_adversary(instances)


class ImprovedLearner(ImprovedPredictor):

    def __init__(self):
        ImprovedPredictor.__init__(self)

    def improve(self, instances):
        X = instances
        I_bad = [x for x in instances if self.initial_learner.predict([x])[0] == 1]
        N = []
        while True:
            new = []
            for instance in I_bad:
                transform_instance = self.adversary.change_instances([instance])[0]
                new_instance = True
                for old_instance in N:
                    if fv_equals(transform_instance.get_feature_vector(),
                                 old_instance.get_feature_vector()):
                        new_instance = False
                if new_instance:
                    new.append(transform_instance)
                    N.append(transform_instance)
            if len(new) == 0:
                break
            self.initial_learner.train(X+N)
            self.adversary.set_adversarial_params(self.initial_learner, X+N)
            break

    def decision_function(self, instances):
        return self.initial_learner.get_model().decision_function_adversary(instances)

