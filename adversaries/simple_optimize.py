import math
from adversaries.adversary import AdversaryStrategy
from data_reader.input import Instance, FeatureVector
from typing import List, Dict
from learners.learner import InitialPredictor
from learners.models.model import BaseModel
from copy import deepcopy

'''Simple optimization of adversarial instance.

Concept:
	Iterates through features in the adversarial instance, flipping features
	that lower the probability of being classified adversarial.
'''

class Adversary(AdversaryStrategy):

	def __init__(self):
		self.lambda_val = -100        # type: float
		self.learn_model = None       # type: BaseModel
		self.num_features = None      # type: int

	def change_instances(self, instances: List[Instance]) -> List[Instance]:
		transformed_instances = []
		for instance in instances:
			transformed_instance = deepcopy(instance)
			if instance.get_label() == 1:
				transformed_instances.append(self.optimize(transformed_instance))
			else:
				transformed_instances.append(transformed_instance)
		return transformed_instances

	def set_params(self, params: Dict):
		if params['lambda_val'] is not None:
			self.lambda_val = params['lambda_val']

	def get_available_params(self) -> Dict:
		params = {'lambda_val': self.lambda_val}
		return params

	def set_adversarial_params(self, learner: InitialPredictor, train_instances: List[Instance]):
		self.learn_model = learner.get_model()
		self.num_features = train_instances[0].get_feature_vector().get_feature_count()

	def optimize(self, instance: Instance):
		for i in range(0, self.num_features):
			orig_prob = self.learn_model.predict_proba_adversary(instance)
			instance.get_feature_vector().flip_bit(i)
			new_prob = self.learn_model.predict_proba_adversary(instance)
			if new_prob >= (orig_prob+math.exp(self.lambda_val)):
				instance.get_feature_vector().flip_bit(i)

		return instance
