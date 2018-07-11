# iterative_retraining_learner.py
# A learner that iteratively retrains and removes outliers based on loss.
# Matthew Sedam

from adlib.learners import Learner
from adlib.learners import TRIMLearner
from adlib.utils.common import logistic_loss
from copy import deepcopy
from data_reader.binary_input import Instance
from typing import Dict, List
import numpy as np


class IterativeRetrainingLearner(Learner):
    """
    A learner that iteratively retrains and removes outliers based on loss.
    """

    def __init__(self, training_instances: List[Instance], verbose=False):
        """
        :param training_instances: the list of training instances
        :param verbose: if True, the learner will print progress
        """

        Learner.__init__(self)
        self.set_training_instances(training_instances)
        self.verbose = verbose

        self.learner = TRIMLearner(self.training_instances,
                                   int(0.75 * len(self.training_instances)),
                                   verbose=self.verbose)
        self.loss = None
        self.loss_threshold = None

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        self.learner.set_training_instances(self.training_instances)
        self.learner.train()
        self.loss = logistic_loss(self.training_instances, self.learner)

        sorted_loss = self.loss[:]
        sorted_loss.sort()
        step_size = 10 * np.mean(np.array(
            list(filter(lambda x: x > 0, sorted_loss[1:] - sorted_loss[:-1]))))
        max_loss_threshold = np.max(self.loss)

        loss_threshold_list = []
        loss_list = []
        learner_list = []

        if self.verbose:
            print('Minimum loss threshold:', np.min(self.loss) + step_size,
                  '\nMaximum loss threshold:', max_loss_threshold,
                  '\nStep size:', step_size)

        self.loss_threshold = np.min(self.loss) + step_size

        while self.loss_threshold < max_loss_threshold:
            training_instances = self.training_instances[:]
            try:
                self._train_helper()
            except:
                self.training_instances = training_instances
                if self.verbose:
                    print('\nLoss threshold:', self.loss_threshold,
                          '- FAILURE\n')
                self.loss_threshold += step_size
                continue

            self.learner.n = len(self.training_instances)
            self.training_instances = training_instances

            loss = sum(self.loss)

            if self.verbose:
                print('\nLoss threshold:', self.loss_threshold, '- loss:', loss,
                      '\n')

            loss_threshold_list.append(self.loss_threshold)
            loss_list.append(loss)
            learner_list.append(deepcopy(self.learner))

            self.loss_threshold += step_size

        # Find index of smallest change in loss
        loss_list = np.array(loss_list)
        loss_derivative = loss_list[1:] - loss_list[:-1]
        loss_derivative /= step_size
        loss_derivative = np.abs(loss_derivative)
        loss_derivative = list(enumerate(loss_derivative))
        loss_derivative.sort(key=lambda x: x[1])

        index = loss_derivative[0][0]
        index = index if loss_list[index] <= loss_list[index + 1] else index + 1

        self.loss_threshold = loss_threshold_list[index]
        self.learner = learner_list[index]
        self.set_training_instances(self.learner.training_instances)

    def _train_helper(self):
        """
        Helper function for self.train()
        """

        self.learner.set_training_instances(self.training_instances)
        self.learner.train()
        self.loss = logistic_loss(self.training_instances, self.learner)

        iteration = 0
        old_training_instances = []
        while set(old_training_instances) != set(self.training_instances):
            old_training_instances = self.training_instances[:]
            instances = []
            for i, inst in enumerate(self.training_instances):
                if self.loss[i] < self.loss_threshold:
                    instances.append(inst)

            self.set_training_instances(instances)

            if self.verbose:
                print('Iteration:', iteration, '- number of instances:',
                      len(self.training_instances))

            self.learner.set_training_instances(self.training_instances)
            self.learner.train()
            self.loss = logistic_loss(self.training_instances, self.learner)

            iteration += 1

    def predict(self, instances):
        return self.learner.predict(instances)

    def set_params(self, params: Dict):
        if params['training_instances'] is not None:
            self.set_training_instances(params['training_instances'])
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.learner = TRIMLearner(self.training_instances,
                                   int(0.75 * len(self.training_instances)),
                                   verbose=self.verbose)
        self.loss = None
        self.loss_threshold = None

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return self.learner.decision_function(X)
