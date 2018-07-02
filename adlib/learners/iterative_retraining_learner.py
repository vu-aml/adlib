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

    def __init__(self, lnr: TRIMLearner, training_instances: List[Instance],
                 verbose=False):
        """
        :param lnr: the base learner
        :param training_instances: the list of training instances
        :param verbose: if True, the learner will print progress
        """

        Learner.__init__(self)
        self.learner = deepcopy(lnr)
        self.learner.set_training_instances(training_instances)
        self.set_training_instances(training_instances)
        self.verbose = verbose

        self.loss = None
        self.loss_threshold = None

    def train(self):
        self.learner.set_training_instances(self.training_instances)
        self.learner.train()
        self.loss = logistic_loss(self.training_instances, self.learner)

        step_size = np.min(self.loss[1:] - self.loss[:-1])
        max_loss_threshold = np.max(self.loss)
        best_loss_threshold = np.min(self.loss) + step_size
        best_learner = deepcopy(self.learner)
        best_loss = None

        self.loss_threshold = best_loss_threshold

        while self.loss_threshold < max_loss_threshold:
            training_instances = self.training_instances[:]
            try:
                self._train_helper()
            except:
                self.training_instances = training_instances
                continue

            self.training_instances = training_instances

            loss = sum(self.loss_threshold)

            if self.verbose:
                print('\nLoss threshold:', self.loss_threshold, '- loss:',
                      self.loss, '\n')

            if not best_loss or loss < best_loss:
                best_loss_threshold = self.loss_threshold
                best_loss = loss
                best_learner = deepcopy(self.learner)

            self.loss_threshold += step_size

        self.loss_threshold = best_loss_threshold
        self.learner = best_learner
        self.set_training_instances(self.learner.training_instances)

    def _train_helper(self):
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

    def predict(self, instances):
        return self.learner.predict(instances)

    def set_params(self, params: Dict):
        if params['lnr'] is not None:
            self.learner = deepcopy(params['lnr'])
        if params['training_instances'] is not None:
            self.set_training_instances(params['training_instances'])
            self.learner.set_training_instances(params['training_instances'])
            self.learner.train()
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.loss = None
        self.loss_threshold = None

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return self.learner.decision_function(X)
