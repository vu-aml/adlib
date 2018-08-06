from typing import List, Dict
from adversaries.adversary import Adversary
from data_reader.binary_input import Instance, BinaryFeatureVector
from learners.learner import learner
from copy import deepcopy
from random import shuffle
from itertools import filterfalse

'''Good Word Attack based on Good Word Attacks on Statistical Spam Filters by 
   Daniel Lowd and Christopher Meek.

Concept:
   This algorithm tries to measure the weight of each words in the email lists and 
   attempts to create a list of n good words. The first-n-words and best-n-words are 
   two methods of discovering the list.
'''

class GoodWord(Adversary):

    BEST_N = 'best_n'
    FIRST_N = 'first_n'

    def __init__(self, n = 100, attack_model_type = BEST_N):
        """
        :param n: number of words needed
        :param attack_model_type: choose the best-n or first-n algorithm
        """
        self.learn_model = None
        self.positive_instance = None    # type: Instance
        self.negative_instance = None    # type: Instance
        self.n = n
        self.num_queries = 0
        self.attack_model_type = attack_model_type
        self.train_instances = None

    def attack(self, instances: List[Instance]) -> List[Instance]:
        word_indices = self.get_n_words()
        transformed_instances = []

        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == learner.positive_classification:
                transformed_instances.append(
                    self.add_words_to_instance(transformed_instance, word_indices))
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

    # throw error for unrecognized parameters?
    def set_params(self, params: Dict):
        if 'n' in params:
            self.n = params['n']
        if 'attack_model_type' in params:
            if not self.is_valid_attack_model_type(params['attack_model_type']):
                raise ValueError('Invalid attack model type')
            self.attack_model_type = params['attack_model_type']

    def is_valid_attack_model_type(self, model_type):
        return (model_type in [GoodWord.BEST_N, GoodWord.FIRST_N])



    def set_pos_neg_instance(self):
        pos_list = [x for x in self.train_instances if x.get_label() == learner.positive_classification]
        shuffle(pos_list)
        neg_list = [x for x in self.train_instances if x.get_label() == learner.negative_classification]
        shuffle(neg_list)
        self.positive_instance = pos_list[0]
        self.negative_instance = neg_list[0]



    def set_adversarial_params(self, learner, train_instances):
        self.learn_model = learner
        instances = train_instances # type: List[Instance]
        self.train_instances = instances

        self.set_pos_neg_instance()

        self.feature_space = set()
        for instance in train_instances:
          self.feature_space.update(instance.get_feature_vector())

    # This is a uniform adversarial cost function, should we add a weight parameter?
    def feature_difference(self, y: BinaryFeatureVector, xa: BinaryFeatureVector) -> List:
        y_array = y.get_csr_matrix()
        xa_array = xa.get_csr_matrix()

        C_y = (y_array - xa_array).indices

        return C_y

    def add_words_to_instance(self, instance, word_indices):
        if word_indices is None:
            return instance
        feature_vector = instance.get_feature_vector()
        for index in word_indices:
            if index not in feature_vector:
                feature_vector.flip_bit(index)
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
            learner.positive_classification):

            prev_message = deepcopy(curr_message)
            word_removed = False
            for index in curr_message:
                if index not in spam_message_words:
                    curr_message.flip_bit(index)
                    word_removed = True
                    break
            if word_removed: continue

            word_added = False
            for index in spam_message:
                if index not in curr_message_words:
                    curr_message.flip_bit(index)
                    curr_message_words.add(index)
                    word_added = True
                    break
            # curr_message and prev_message will not change for any more iterations
            if not word_added:
                print('Could not find witness')
                return None
        return (curr_message, prev_message)

    def first_n_words(self, spam_message, legit_message):
        if not self.n: raise ValueError('Must specify n')

        negative_weight_word_indices = set()
        return_message = self.find_witness()
        count = 1
        while return_message is None and count <= 20:
            self.set_pos_neg_instance()
            count += 1
            return_message = self.find_witness()
        if return_message is None:
            print("Cannot find witness after a few iterations")
            print("Attack fails")
            return None

        spam_message  = return_message[0]
        # use the feature vector of the negative instance just to iterate over all the indices in a
        # feature vector, the actual values do not matter

        # this doesn't iterate over all possible features because of the current feature vector
        # implementation
        for feature in self.feature_space:
            if spam_message.get_feature(feature) == 0:
                spam_message.flip_bit(feature)
                prediction_result = self.predict_and_record(spam_message)
                if prediction_result == learner.negative_classification:
                    negative_weight_word_indices.add(feature)
                if len(negative_weight_word_indices) == self.n:
                    return negative_weight_word_indices
                # remove word from message so spam_message stays the same for each iteration
                spam_message.flip_bit(feature)
        return negative_weight_word_indices

    def best_n_words(self, spam_message, legit_message):
        return_message = self.find_witness()
        count = 1
        while return_message is None and count <= 20:
            self.set_pos_neg_instance()
            count += 1
            return_message = self.find_witness()
        if return_message is None:
            print("Cannot find witness")
            print("Attack fails")
            return None

        barely_spam_message, barely_legit_message = return_message[0], return_message[1]
        positive_weight_word_indices = self.build_word_set(barely_legit_message, learner.positive_classification)
        negative_weight_word_indices = self.build_word_set(barely_spam_message, learner.negative_classification)
        best_n_word_indices = set()
        iterations_without_change = 0
        max_iterations_without_change = 10
        for spammy_word_index in positive_weight_word_indices:
            is_index_in_spam_msg = barely_spam_message.get_feature(spammy_word_index) == 1
            if not is_index_in_spam_msg: barely_spam_message.flip_bit(spammy_word_index)
            if not is_index_in_spam_msg: barely_spam_message.flip_bit(spammy_word_index)
            small_weight_word_indices = self.build_word_set(
                barely_spam_message,
                learner.positive_classification,
                negative_weight_word_indices
            )
            large_weight_word_indices = self.build_word_set(
                barely_spam_message,
                learner.negative_classification,
                negative_weight_word_indices
            )
            if not is_index_in_spam_msg: barely_spam_message.flip_bit(spammy_word_index)

            if len(best_n_word_indices) + len(large_weight_word_indices) < self.n:
                negative_weight_word_indices = negative_weight_word_indices - large_weight_word_indices
                best_n_word_indices = best_n_word_indices.union(large_weight_word_indices)
                if len(large_weight_word_indices) == 0:
                    iterations_without_change += 1
                else:
                    iterations_without_change = 0
            else:
                negative_weight_word_indices = negative_weight_word_indices - small_weight_word_indices
                if len(small_weight_word_indices) == 0:
                    iterations_without_change += 1
                else:
                    iterations_without_change = 0

            if iterations_without_change == max_iterations_without_change:
                for i in range(min(self.n - len(best_n_word_indices), len(negative_weight_word_indices))):
                    best_n_word_indices.add(negative_weight_word_indices.pop())
                return best_n_word_indices
        return best_n_word_indices

    def build_word_set(self, message, intended_classification, indices_to_check = None):
        # if no specific indices are passed in, defaults to checking every index
        # build list of words by adding dictionary word to the message and sending it through
        # the filter
        indices_to_check = indices_to_check if indices_to_check != None else self.feature_space
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
        return self.predict(Instance(0, message))

    def predict(self, instance):
        return self.learn_model.predict(instance)

    def get_n_words(self):
        # identify the moel type and get words for attack procedure
        if self.attack_model_type == GoodWord.FIRST_N:
            return self.first_n_words(
                self.positive_instance.get_feature_vector(),
                self.negative_instance.get_feature_vector()
            )
        if self.attack_model_type == GoodWord.BEST_N:
            return self.best_n_words(
                self.positive_instance.get_feature_vector(),
                self.negative_instance.get_feature_vector()
            )
        else:
            raise ValueError('Unknown attack model type')
