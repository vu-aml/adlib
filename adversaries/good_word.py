from typing import List, Dict
from adversaries.adversary import Adversary
from data_reader.input import Instance, FeatureVector
from learners.learner import InitialPredictor
from data_reader import operations
from copy import deepcopy
from itertools import filterfalse

'''Good Word Attack.

Concept:
'''

class GoodWord(Adversary):

    BEST_N = 'best_n'
    FIRST_N = 'first_n'

    def __init__(self):
        self.learn_model = None                # type: InitialPredictor
        self.positive_instance = None    # type: Instance
        self.negative_instance = None    # type: Instance
        self.n = None
        self.num_queries = 0

    def attack(self, instances: List[Instance]) -> List[Instance]:
        word_indices = self.get_n_words()
        transformed_instances = []

        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == InitialPredictor.positive_classification:
                transformed_instances.append(
                    self.add_words_to_instance(transformed_instance, word_indices)
                )
            else:
                transformed_instances.append(transformed_instance)
        print('Number of queries issued:', self.num_queries)
        return transformed_instances

    def get_available_params(self):
        return {
            'n': self.n,
            'positive_instance': self.positive_instance,
            'negative_instance': self.negative_instance,
        }

    def set_params(self, params: Dict):
        if 'n' in params:
            self.n = params['n']
        if 'attack_model_type' in params:
            self.attack_model_type = params['attack_model_type']
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
        self.feature_space = set()
        for instance in train_instances:
          self.feature_space.update(instance.get_feature_vector())

    # This is a uniform adversarial cost function, should we add a weight parameter?
    def feature_difference(self, y: FeatureVector, xa: FeatureVector) -> List:
        y_array = y.get_csr_matrix()
        xa_array = xa.get_csr_matrix()

        C_y = (y_array - xa_array).indices

        return C_y

    def add_words_to_instance(self, instance, word_indices):
        for index in word_indices:
            instance.get_feature_vector().flip_bit(index)
        return instance

    # Find a spam and legit message that only differ by 1 word
    def find_witness(self):
        curr_message = deepcopy(self.negative_instance.get_feature_vector())
        curr_message_words = set(curr_message)
        spam_message = self.positive_instance.get_feature_vector()
        spam_message_words = set(spam_message)
        prev_message = None
        # loop until current message is classified as spam
        while (self.predict_and_record(curr_message) !=
            InitialPredictor.positive_classification):

            prev_message = deepcopy(curr_message)
            word_removed = False
            for index in curr_message:
                if index in spam_message_words:
                    curr_message.flip_bit(index)
                    word_removed = True
                    break
            if not word_removed:
                for index in spam_message:
                    if index not in curr_message_words:
                        curr_message.flip_bit(index)
        return (curr_message, prev_message)

    def first_n_words(self, spam_message, legit_message):
        if not self.n: raise ValueError('Must specify n')
        negative_weight_word_indices = set()
        barely_spam_message, barely_legit_message = self.find_witness()
        # use the feature vector of the negative instance just to iterate over all the indices in a
        # feature vector, the actual values do not matter

        # this doesn't iterate over all possible features because of the current feature vector
        # implementation
        for feature in self.feature_space:
            if spam_message.get_feature(feature) == 0:
                spam_message.flip_bit(feature)
                prediction_result = self.predict_and_record(spam_message)
                if prediction_result == InitialPredictor.negative_classification:
                    negative_weight_word_indices.add(feature)
                if len(self.negative_instance.get_feature_vector()) == self.n:
                    return negative_weight_word_indices
                # remove word from message so spam_message stays the same for each iteration
                spam_message.flip_bit(feature)
        return negative_weight_word_indices

    def best_n_words(self, spam_message, legit_message):
        barely_spam_message, barely_legit_message = self.find_witness()
        positive_weight_word_indices = self.build_word_set(barely_legit_message, InitialPredictor.positive_classification)
        negative_weight_word_indices = self.build_word_set(barely_spam_message, InitialPredictor.negative_classification)
        best_n_word_indices = set()
        iterations_without_change = 0
        max_iterations_without_change = 10
        for spammy_word_index in positive_weight_word_indices:
            barely_spam_message.flip_bit(spammy_word_index),
            small_weight_word_indices = self.build_word_set(
                barely_spam_message,
                InitialPredictor.positive_classification,
                negative_weight_word_indices
            )
            large_weight_word_indices = self.build_word_set(
                barely_spam_message,
                InitialPredictor.negative_classification,
                negative_weight_word_indices
            )
            barely_spam_message.flip_bit(spammy_word_index)
            if len(best_n_word_indices) + len(large_weight_word_indices) > self.n:
                negative_weight_word_indices = negative_weight_word_indices - large_weight_word_indices
                best_n_word_indices = best_n_word_indices.union(large_weight_word_indices)
                iterations_without_change = 0
            else:
                negative_weight_word_indices = negative_weight_word_indices - small_weight_word_indices
                iterations_without_change += 1
            if iterations_without_change == max_iterations_without_change:
                for i in range(n - len(best_n_word_indices)):
                    best_n_word_indices.add(negative_weight_word_indices.pop())
                return best_n_word_indices
        return best_n_word_indices

    def build_word_set(self, message, intended_classification, indices_to_check = None):
        # if no specific indices are passed in, defaults to checking every index
        indices_to_check = indices_to_check or self.feature_space
        result = set()
        for index in indices_to_check:
            if message.get_feature(index) == 0:
                message.flip_bit(index)
                prediction_result = self.predict_and_record(message)
                if prediction_result == intended_classification:
                    result.add(index)
                message.flip_bit(index)
        return result

    def predict_and_record(self, message):
        self.num_queries += 1
        return self.learn_model.predict(Instance(0, message))

    def get_n_words(self):
        if self.attack_model_type == Adversary.FIRST_N:
            return self.first_n_words(
                self.positive_instance.get_feature_vector(),
                self.negative_instance.get_feature_vector()
            )
        if self.attack_model_type == Adversary.BEST_N:
            return self.best_n_words(
                self.positive_instance.get_feature_vector(),
                self.negative_instance.get_feature_vector()
            )
        else:
            raise ValueError('Unknown attack model type')
