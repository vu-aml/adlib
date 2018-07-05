# dp_learner_test.py
# Tests data poisoning learners
# Matthew Sedam. 2018.

from adlib.learners import SimpleLearner
from adlib.adversaries.label_flipping import LabelFlipping
from adlib.adversaries.k_insertion import KInsertion
from adlib.adversaries.datamodification.data_modification import DataModification
from adlib.tests.adversaries.data_modification_test import calculate_target_theta
from adlib.utils.common import calculate_correct_percentages
from copy import deepcopy
from data_reader.binary_input import Instance
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from sklearn import svm
from typing import Dict, List
import numpy as np
import time


class TestDataPoisoningLearner:

    def __init__(self, learner_name: str,
                 attacker_name: str,
                 dataset: EmailDataset,
                 params: Dict = None,
                 verbose=True):
        """
        Test setup.
        :param learner_name: Either 'trim', 'atrim', or 'irl'
        :param attacker_name: Either 'label-flipping', 'k-insertion', 'data-modification', or
                              'dummy'
        :param params: the params to pass to the learner - if None, defaults will be used
        :param verbose: if True, will print START and STOP and set learners and attackers to
                        verbose mode
        """

        if learner_name.lower() not in ['trim', 'atrim', 'irl']:
            raise ValueError('Learner name not trim, atrim, nor irl.')

        if attacker_name.lower() not in ['label-flipping', 'k-insertion',
                                         'data-modification', 'dummy']:
            raise ValueError('Attacker name not label-flipping, k-insertion, '
                             'data-modification, nor dummy.')

        self.learner_name = learner_name.lower()
        if self.learner_name == 'trim':
            self.learner_name = 'TRIM Learner'
        elif self.learner_name == 'atrim':
            self.learner_name = 'Alternating TRIM Learner'
        else:
            self.learner_name = 'Iterative Retraining Learner'

        self.attacker_name = attacker_name.lower()
        self.params = params
        self.verbose = verbose

        training_data, testing_data = dataset.split({'train': 25, 'test': 75})
        self.training_instances = load_dataset(training_data)
        self.testing_instances = load_dataset(testing_data)

        self.learner = None
        self.attack_learner = None
        self.attacker = None
        self.attack_instances = None
        self.training_pred_labels = None
        self.testing_pred_labels = None

    def test(self):
        if self.verbose:
            print()
            print('###################################################', end='')
            print('################\nSTART ', self.learner_name, ' test.\n', sep='')

        begin = time.time()

        if self.verbose:
            print('Training sample size: ', len(self.training_data), '/400\n',
                  sep='')

        # Setting the default learner
        learning_model = svm.SVC(probability=True, kernel='linear')
        self.learner = SimpleLearner(learning_model, self.training_data)
        self.learner.train()

        self.training_pred_labels = self.learner.predict(self.training_instances)
        self.testing_pred_labels = self.learner.predict(self.testing_instances)

        # Execute the attack
        if self.attacker_name == 'label-flipping':
            cost = list(np.random.binomial(2, 0.5, len(self.training_data)))
            total_cost = 80  # flip around 80 labels
            if self.params:
                self.attacker = LabelFlipping(deepcopy(self.learner), **self.params)
            else:
                self.attacker = LabelFlipping(deepcopy(self.learner), cost, total_cost,
                                              verbose=self.verbose)
        elif self.attacker_name == 'k-insertion':
            self.attacker = KInsertion(deepcopy(self.learner),
                                       self.training_instances[0],
                                       number_to_add=100,  # 100 / (400 + 100) = 20%
                                       verbose=self.verbose)
        elif self.attacker_name == 'data-modification':
            target_theta = calculate_target_theta(deepcopy(self.learner),
                                                  deepcopy(self.training_instances),
                                                  deepcopy(self.testing_instances))

            self.attacker = DataModification(deepcopy(self.learner), target_theta,
                                             verbose=self.verbose)
        else:  # self.attacker_name == 'dummy'
            class DummyAttacker:
                def attack(self, instances):
                    attack_instances = deepcopy(instances)
                    tmp = np.random.binomial(1, 0.8, 100)
                    for i, val in enumerate(tmp):
                        if val == 1:
                            attack_instances[i].set_label(attack_instances[i].get_label() * -1)
                    return attack_instances

            self.attacker = DummyAttacker()

        if self.verbose:
            print('###################################################', end='')
            print('################\nSTART ', self.attacker_name, ' attack.\n', sep='')

            self.attack_instances = self.attacker.attack(deepcopy(self.training_instances))

            print('\nEND', self.attacker_name, 'attack.')
            print('###################################################', end='')
            print('################')
            print()

        # Retrain the model with poisoned data
        learning_model = svm.SVC(probability=True, kernel='linear')
        self.attack_learner = SimpleLearner(learning_model, self.attack_instances)
        self.attack_learner.train()
