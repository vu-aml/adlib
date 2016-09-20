from learners.learner import InitialPredictor, ImprovedPredictor
from typing import List
from types import FunctionType
import numpy as np
from adversaries.nash_equilibrium import Adversary, predict_instance
from data_reader.input import Instance, FeatureVector


# def predict_instance(fv: np.array, a_learner: np.array) -> float:
# 	fv = np.dot(a_learner, fv)
# 	return fv

"""Companion class to adversary Nash Equilibrium.

Concept:
	Learner trains (in the usual way) and defines its loss function.
	Improved learner gathers equilibrium solution weight vector (as determined
	by adversary) and uses it to predict on the transformed data.

"""

class Learner(InitialPredictor):

	def __init__(self):
		InitialPredictor.__init__(self)

	def loss_function(self, z: float, y: int):
		return np.log(1 + np.exp(-1 * y * z))

	def get_loss_function(self) -> FunctionType:
		return getattr(self, 'loss_function')


class ImprovedLearner(ImprovedPredictor):

	def __init__(self):
		ImprovedPredictor.__init__(self)
		self.weight_vector = None

	def predict(self, instances: List[Instance]):
		predictions = []
		for instance in instances:
			features = instance.get_feature_vector().get_csr_matrix().toarray()[0]
			predictions.append(np.sign(predict_instance(features, self.weight_vector)))
		return predictions

	def set_adversarial_params(self, learner: Learner, adversary: Adversary):
		self.weight_vector = adversary.get_learner_params()