from typing import List, Dict
from adversaries.adversary import AdversaryStrategy
from data_reader.input import Instance, FeatureVector
from learners.learner import InitialPredictor
from data_reader import operations
from copy import deepcopy
from itertools import filterfalse

'''Good Word Attack.

Concept:
'''

class Adversary(AdversaryStrategy):

	def __init__(self):
		self.learn_model = None        # type: InitialPredictor
		self.positive_instance = None  # type: Instance
		self.negative_instance = None  # type: Instance

	def change_instances(self, instances: List[Instance]) -> List[Instance]:
		transformed_instances = []

		for instance in instances:
			transformed_instance = deepcopy(instance)
			if instance.get_label() == InitialPredictor.positive_classification:
				transformed_instances.append(self.find_boolean_IMAC(transformed_instance))
			else:
				transformed_instances.append(transformed_instance)
		return transformed_instances

	def get_available_params(self):
		return None

	def set_params(self, params: Dict):
		if params[''] is not None:
			self.n = params['n']
    else:
      raise ValueError('Must specify n')

		return None

	def set_adversarial_params(self, learner, train_instances):
		self.learn_model = learner
		instances = train_instances # type: List[Instance]
		self.positive_instance = next(
      (x for x in instances if x.get_label() == InitialPredictor.positive_classification),
      None
    )
		self.negative_instance = next(
      (x for x in instances if x.get_label() == InitialPredictor.negative_classification),
      None
    )

  # This is a uniform adversarial cost function, should we add a weight parameter?
	def feature_difference(self, y: FeatureVector, xa: FeatureVector) -> List:
		y_array = y.get_csr_matrix()
		xa_array = xa.get_csr_matrix()

		C_y = (y_array - xa_array).indices

		return C_y

  # Find a spam and legit message that only differ by 1 word
  def find_witness(self):
    curr_message = deepcopy(self.negative_instance.get_feature_vector())
    curr_message_words = set(curr_message)
    spam_message = self.positive_instance.get_feature_vector()
    # loop until current message is classified as spam
    while (self.learn_model.predict(Instance(0,curr_message)) !=
      InitialPredictor.positive_classification):

      prev_message = deepcopy(curr_message)
      word_removed = False
      for index in range(len(curr_message)):
        # if the word occurs in curr_message but not in spam_message
        if if curr_message[index] == 1 and spam_message[index] == 0:
          # remove word from curr_message
          curr_message.flip_bit(index)
          word_removed = True
          break
      if not word_removed:
        for index in range(len(spam_message)):
          if curr_message[index] == 0 and spam_message[index] == 1:
            curr_message.flip_bit(index)
    return (curr_message, prev_message)

  def first_n_words(self, spam_message, legit_message):
    if not self.n: raise ValueError('Must specify n')
    negative_weight_word_indices = set()
    barely_spam_message, barely_legit_message = self.find_witness(spam_message, legit_message)
    # use the feature vector of the negative instance just to iterate over all the indices in a
    # feature vector, the actual values do not matter
    for index in self.get_all_word_indices():
      if spam_message[index] == 0:
        spam_message.flip_bit(index)
        prediction_result = self.learn_model.predict(Instance(0, spam_message))
        if prediction_result == InitialPredictor.negative_classification:
          negative_weight_word_indices.add(index)
        if negative_instance.size() == self.n:
          return negative_weight_word_indices
        # remove word from message so spam_message stays the same for each iteration
        spam_message.flip_bit(index)
    return negative_weight_word_indices

  def best_n_words(self, spam_message, legit_message):
    barely_spam_message, barely_legit_message = self.find_witness(spam_message, legit_message)
    positive_weight_word_indices = self.build_word_set(legit_message, InitialPredictor.positive_classification)
    negative_weight_word_indices = self.build_word_set(spam_message, InitialPredictor.negative_classification)
    best_n_word_indices = set()
    iterations_without_change = 0
    max_iterations_without_change = 10
    for spammy_word_index in positive_weight_word_indices:
      small_weight_word_indices = self.build_word_set(
        barely_spam_message.flip_bit(spammy_word_index),
        InitialPredictor.positive_classification,
        negative_weight_word_indices
      )
      large_weight_word_indices = self.build_word_set(
        barely_spam_message.flip_bit(spammy_word_index),
        InitialPredictor.negative_classification,
        negative_weight_word_indices
      )
      if best_n_word_indices.size() + large_weight_word_indices.size() > self.n:
        negative_weight_word_indices = negative_weight_word_indices - large_weight_word_indices
        best_n_word_indices = best_n_word_indices.union(large_weight_word_indices)
        iterations_without_change = 0
      else:
        negative_weight_word_indices = negative_weight_word_indices - small_weight_word_indices
        iterations_without_change += 1
      if iterations_without_change == max_iterations_without_change:
        for i in range(n - best_n_word_indices.size()):
          best_n_word_indices.add(negative_weight_word_indices.pop())
        return best_n_word_indices
    return best_n_word_indices

  def build_word_set(self, message, intended_classification, indices_to_check = None):
    # if no specific indices are passed in, defaults to checking every index
    indices_to_check = indices_to_check or self.get_all_word_indices()
    result = set()
    for index in indices_to_check:
      if message[index] == 0:
        message.flip_bit(index)
        prediction_result = self.learn_model.predict(Instance(0, spam_message))
        if prediction_result == intended_classification:
          result.add(index)
        message.flip_bit(index)
    return result

  def get_all_word_indices(self):
    return range(len(self.negative_instance.get_feature_vector()))
