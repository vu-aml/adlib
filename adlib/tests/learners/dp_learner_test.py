# dp_learner_test.py
# Tests data poisoning learners
# Matthew Sedam. 2018.

from adlib.learners import SimpleLearner
from adlib.learners import TRIMLearner
from adlib.learners import AlternatingTRIMLearner
from adlib.learners import IterativeRetrainingLearner
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
        :param dataset: the dataset
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

        training_data, testing_data = dataset.split({'train': 50, 'test': 50})
        self.training_instances = load_dataset(training_data)
        self.testing_instances = load_dataset(testing_data)

        self.learner = None  # SVM with clean dataset
        self.attack_learner = None  # SVM with attacked dataset
        self.dp_learner = None  # Learner we are testing
        self.attacker = None  # the attacker
        self.attack_instances = None  # the attacked instances

        # Before attack
        self.training_pred_labels = None  # the predicted labels of the training set for the SVM
        self.testing_pred_labels = None  # the predicted labels of the testing set for the SVM

        # After attack
        self.attack_training_pred_labels = None  # attacker predicted labels for training set SVM
        self.attack_testing_pred_labels = None  # attacker predicted labels for the testing set SVM
        self.dp_learner_training_pred_labels = None  # predicted labels for training set DP Learner
        self.dp_learner_testing_pred_labels = None  # predicted labels for the training set DP L.

        self.labels = []  # true labels
        for inst in self.training_instances + self.testing_instances:
            self.labels.append(inst.get_label())

    def test(self):
        if self.verbose:
            print('\n###################################################################')
            print('START', self.learner_name, 'test.\n')

        self._setup()
        self._attack()
        self._retrain()

        begin = time.time()
        self._run_learner()
        end = time.time()

        if self.verbose:
            print('\nEND', self.learner_name, 'test.')
            print('###################################################################\n')

        return (list(self.labels),
                list(self.training_pred_labels) + list(self.testing_pred_labels),
                list(self.attack_training_pred_labels) + list(self.attack_testing_pred_labels),
                list(self.dp_learner_training_pred_labels) +
                list(self.dp_learner_testing_pred_labels),
                end - begin)

    def _setup(self):
        if self.verbose:
            print('Training sample size: ', len(self.training_instances), '/400\n',
                  sep='')

        # Setting the default learner
        learning_model = svm.SVC(probability=True, kernel='linear')
        self.learner = SimpleLearner(learning_model, self.training_instances)
        self.learner.train()

        self.training_pred_labels = self.learner.predict(self.training_instances)
        self.testing_pred_labels = self.learner.predict(self.testing_instances)

    def _attack(self):
        # Execute the attack
        if self.attacker_name == 'label-flipping':
            cost = list(np.random.binomial(2, 0.5, len(self.training_instances)))
            total_cost = 40  # flip around 40 labels
            if self.params:
                self.attacker = LabelFlipping(deepcopy(self.learner), **self.params)
            else:
                self.attacker = LabelFlipping(deepcopy(self.learner), cost, total_cost,
                                              verbose=self.verbose)
        elif self.attacker_name == 'k-insertion':
            self.attacker = KInsertion(deepcopy(self.learner),
                                       self.training_instances[0],
                                       number_to_add=50,  # 50 / (200 + 50) = 20%
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
                    tmp = np.random.binomial(1, 0.2, 200)
                    for i, val in enumerate(tmp):
                        if val == 1:
                            attack_instances[i].set_label(attack_instances[i].get_label() * -1)
                    return attack_instances

            self.attacker = DummyAttacker()

        if self.verbose:
            print('\n###################################################################')
            print('START', self.attacker_name, 'attack.\n')

        if self.attacker_name == 'data-modification':
            self.attack_instances = self.attacker.attack(deepcopy(self.training_instances[:40]))
            self.attack_instances += deepcopy(self.training_instances[:-160])
        else:
            self.attack_instances = self.attacker.attack(deepcopy(self.training_instances))

        if self.verbose:
            print('\nEND', self.attacker_name, 'attack.')
            print('###################################################################\n')

    def _retrain(self):
        # Retrain the model with poisoned data
        learning_model = svm.SVC(probability=True, kernel='linear')
        self.attack_learner = SimpleLearner(learning_model, self.attack_instances)
        self.attack_learner.train()

        self.attack_training_pred_labels = self.attack_learner.predict(self.training_instances)
        self.attack_testing_pred_labels = self.attack_learner.predict(self.testing_instances)

    def _run_learner(self):
        if self.verbose:
            print('\n###################################################################')
            print('START ', self.learner_name, '.\n', sep='')

        if self.learner_name == 'TRIM Learner':
            self.dp_learner = TRIMLearner(deepcopy(self.attack_instances),
                                          int(len(self.attack_instances) * 0.8),
                                          verbose=self.verbose)
        elif self.learner_name == 'Alternating TRIM Learner':
            self.dp_learner = AlternatingTRIMLearner(deepcopy(self.attack_instances),
                                                     verbose=self.verbose)
        else:  # self.learner_name == 'Iterative Retraining Learner'
            self.dp_learner = IterativeRetrainingLearner(deepcopy(self.attack_instances),
                                                         verbose=self.verbose)

        self.dp_learner.train()

        if self.verbose:
            print('\nEND ', self.learner_name, '.', sep='')
            print('###################################################################\n')

        self.dp_learner_training_pred_labels = self.dp_learner.predict(self.training_instances)
        self.dp_learner_testing_pred_labels = self.dp_learner.predict(self.testing_instances)
