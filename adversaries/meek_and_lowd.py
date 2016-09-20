from typing import List, Dict
from adversaries.adversary import AdversaryStrategy
from data_reader.input import Instance, FeatureVector
from learners.learner import InitialPredictor
from data_reader import operations
from copy import deepcopy
from itertools import filterfalse
from math import sqrt

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

  # This is a uniform adversarial cost function, should we add a weight parameter?
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

  # W: list of search vectors of unit cost
  # C_plus = initial lower bound on cost
  # C_minus = initial upper bound on cost
  def multi_line_search(self, W, cost_plus, cost_minus, instance, epsilon):
		x_minus = self.negative_instance.get_feature_vector()
    x_star = deepcopy(x_minus)
    # x_a subset of positive instances
    x_a = instance.get_feature_vector()
    # Don't think I acutally need t
    t = 0
    while cost_minus / cost_plus > 1 + epsilon:
      cost_t = math.sqrt(cost_plus * cost_minus)
      is_negative_vertex_found = False
      for e in W:
        # looks like we're assuming that y and e have the same dimensions
        if self.learn_model.predict(Instance(0, y + cost_t * e)) == -1:
          x_star = y + cost_t * e
          is_negative_vertex_found = True
          # Prune all costs that result in positive prediction
          for i in W:
            if self.learn_model.predict(Instance(0, y + cost_t * i)) == 1:
              # deleting from a list when you iterate through it may cause bugs
              W.delete(i)
          break
      cost_next_plus = cost_plus
      cost_next_minus = cost_minus
    if is_negative_vertex_found:
      cost_next_minus = cost_t
    else:
      cost_next_plus = cost_t
    t += 1
    return x_star

  def convex_set_search(self, instance):
    return None

  # TODO: figure out if I need to implement these
  # TODO: fix parameters
  def find_continuous_weights(self, instances):
    return None

  # TODO: fix parameters
  def find_continuous_IMAC(self, instances):
    return None
