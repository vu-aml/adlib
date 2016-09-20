from typing import List, Dict
from adversaries.adversary import AdversaryStrategy
from data_reader.input import Instance, FeatureVector
from learners.learner import InitialPredictor
from data_reader import operations
from copy import deepcopy
from itertools import filterfalse

'''Meek and Lowd IMAC.

Concept:
	Given an ideal adversarial instance A and an instance classified as
	non adversarial X, finds the feature set difference between A and X
	and iteratively applies those features to A, rolling each change back
	if it results in an adversarial classification under the learner's model.
'''

class Adversary(AdversaryStrategy):

	def __init__(self):
		self.learn_model = None        # type: InitialPredictor
		self.positive_instance = None  # type: Instance
		self.negative_instance = None  # type: Instance

	def change_instances(self, instances: List[Instance]) -> List[Instance]:
		'''Change adversarial instances by finding boolean IMAC.

		used to transform all test instances, assuming that each instance classified as
		adversarial is the 'ideal' instance. To find a single IMAC based on a user specified
		adversarial 'ideal' instance, override test data to pass in single (or multiple)
		adversarially preferred instances.

		'ideal' instance refers to the test vector that makes the best argument for the
		adversaries intention. E.g. spam filtering, email that the adversary classifies as being
		most useful to their efforts
		'''
		transformed_instances = []

		for instance in instances:
			transformed_instance = deepcopy(instance)
			if instance.get_label() == 1:
				transformed_instances.append(self.find_boolean_IMAC(transformed_instance))
			else:
				transformed_instances.append(transformed_instance)
		return transformed_instances

	def get_available_params(self):
		return None

	def set_params(self, params: Dict):
		return None

	def set_adversarial_params(self, learner, train_instances):
		self.learn_model = learner
		instances = train_instances # type: List[Instance]
		self.positive_instance = next((x for x in instances if x.get_label() == 1), None)
		self.negative_instance = next((x for x in instances if x.get_label() == -1), None)

	def feature_difference(self, y: FeatureVector, xa: FeatureVector) -> List:
		y_array = y.get_csr_matrix()
		xa_array = xa.get_csr_matrix()

		C_y = (y_array - xa_array).indices

		return C_y

	def find_boolean_IMAC(self, instance):
		x_minus = self.negative_instance.get_feature_vector()
		y = deepcopy(x_minus)
		xa = instance.get_feature_vector()
		while True:
			y_prev = deepcopy(y)
			C_y = self.feature_difference(y, xa)

			for index in C_y:
				y.flip_bit(index)
				if self.learn_model.predict(Instance(0,y)) == 1:
					y.flip_bit(index)

			C_y = self.feature_difference(y, xa)
			not_C_y = list(filterfalse(lambda x: x in C_y, range(0, y.feature_count)))

			for index1 in C_y:
				for index2 in C_y:
					if index2 <= index1: continue
					for index3 in not_C_y:
						y.flip_bit(index1)
						y.flip_bit(index2)
						y.flip_bit(index3)
						if self.learn_model.predict(Instance(0,y)) == 1:
							y.flip_bit(index1)
							y.flip_bit(index2)
							y.flip_bit(index3)

			if operations.fv_equals(y, y_prev):
				return Instance(1, y)
