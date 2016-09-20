from learners.learner import InitialPredictor, ImprovedPredictor
from typing import Dict, List
from types import FunctionType
import numpy as np
from data_reader.input import FeatureVector, Instance
from adversaries.stackelberg import Adversary
from data_reader.operations import sparsify

"""Companion class to adversary Stackelberg equilibrium.

Concept:
	Learner trains (in the usual way) and defines its loss function, costs, and
	density weight. Improved learner gathers equilibrium solution weight vector
	(as determined by adversary) and uses it to predict on the transformed data.

"""

class Learner(InitialPredictor):

	def __init__(self):
		InitialPredictor.__init__(self)
		self.costs = None # type: List[float]
		self.density_weight = 1.0

	def set_params(self, params: Dict):
		if params['costs'] is not None:
			self.costs = params['costs']
		if params['density_weight'] is not None:
			self.density_weight = params['density_weight']
		InitialPredictor.set_params(self, params)

	def get_available_params(self) -> Dict:
		params = InitialPredictor.get_available_params(self)
		params['costs'] = None
		params['density_weight'] = self.density_weight
		return params

	def get_costs(self) -> List[float]:
		return self.costs

	def get_density_weight(self) -> float:
		return self.density_weight

	def loss_function(self, z: float, y: int):
		return np.log(1 + np.exp(-1 * y * z))

	def feature_mapping(self, x: FeatureVector):
		return x.get_csr_matrix().toarray()[0]

	def get_loss_function(self) -> FunctionType:
		return getattr(self, 'loss_function')

	def get_feature_mapping(self) -> FunctionType:
		return getattr(self, 'feature_mapping')


class ImprovedLearner(ImprovedPredictor):

	def __init__(self):
		ImprovedPredictor.__init__(self)
		self.costs = None # type: List[float]
		self.weight_vector = None
		self.feature_mapping = None

	def predict(self, instances: List[Instance]):
		instance_matrix = sparsify(instances)[1]
		predictions = instance_matrix.dot(self.weight_vector)
		predictions = np.sign(predictions)

		return predictions

	def set_adversarial_params(self, learner: Learner, adversary: Adversary):
		self.costs = learner.get_costs()
		self.weight_vector = adversary.get_learner_params()
		self.feature_mapping = learner.get_feature_mapping()
