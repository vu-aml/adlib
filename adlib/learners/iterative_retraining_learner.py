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

        self.lnr = TRIMLearner(self.training_instances,
                               int(0.75 * len(self.training_instances)),
                               verbose=self.verbose)
        self.loss = None
        self.loss_threshold = None
        self.irl_selection = np.full(len(self.training_instances), 1)

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        self.lnr.set_training_instances(self.training_instances)
        self.lnr.train()
        self.lnr.redo_problem_on_train = False
        self.loss = (logistic_loss(self.training_instances, self.lnr) /
                     sum(self.irl_selection))

        sorted_loss = self.loss[:]
        sorted_loss.sort()
        step_size = np.mean(np.array(
            list(filter(lambda x: x > 0, sorted_loss[1:] - sorted_loss[:-1]))))
        max_loss_threshold = np.max(self.loss)
        best_loss_threshold = np.median(self.loss)
        best_lnr = None
        best_loss = None

        if self.verbose:
            print('Minimum loss threshold:', best_loss_threshold,
                  '\nMaximum loss threshold:', max_loss_threshold,
                  '\nStep size:', step_size)

        self.loss_threshold = best_loss_threshold

        while self.loss_threshold < max_loss_threshold:
            self.irl_selection = np.full(len(self.training_instances), 1)
            try:
                self._train_helper()
            except:
                if self.verbose:
                    print('\nLoss threshold:', self.loss_threshold,
                          '- FAILURE\n')
                self.loss_threshold += step_size
                continue

            self.lnr.n = sum(self.irl_selection)
            loss = sum(self.loss)

            if self.verbose:
                print('\nLoss threshold:', self.loss_threshold, '- loss:', loss,
                      '\n')

            if not best_loss or loss < best_loss:
                best_loss_threshold = self.loss_threshold
                best_loss = loss
                best_lnr = deepcopy((self.lnr.training_instances, self.lnr.n,
                                     self.lnr.lda, self.lnr.verbose, self.lnr.w,
                                     self.lnr.b, self.lnr.irl_selection))

            self.loss_threshold += step_size

        self.loss_threshold = best_loss_threshold
        self.lnr = TRIMLearner(best_lnr[0], best_lnr[1], best_lnr[2], best_lnr[3])
        self.lnr.w, self.lnr.b = best_lnr[4], best_lnr[5]
        self.lnr.irl_selection = best_lnr[6]

    def _train_helper(self):
        """
        Helper function for self.train()
        """

        self.lnr.irl_selection = self.irl_selection
        self.lnr.train()
        self.loss = (logistic_loss(self.training_instances, self.lnr) /
                     sum(self.irl_selection))

        iteration = 0
        old_irl_selection = np.full(len(self.irl_selection), -1)
        while np.linalg.norm(self.irl_selection - old_irl_selection) != 0:
            old_irl_selection = deepcopy(self.irl_selection)

            self.irl_selection = np.full(len(self.irl_selection), 0)
            for i, loss in enumerate(self.loss):
                if loss < self.loss_threshold:
                    self.irl_selection[i] = 1

            # Have to have at least 50% of the instances to train with as
            # we assume at least 50% of the data is clean
            if sum(self.irl_selection) < 0.5 * len(self.training_instances):
                raise ValueError()

            if self.verbose:
                print('IRL Iteration:', iteration, '- number of instances:',
                      sum(self.irl_selection))

            self.lnr.irl_selection = self.irl_selection
            self.lnr.train()
            self.loss = (logistic_loss(self.training_instances, self.lnr) /
                         sum(self.irl_selection))

            iteration += 1

    def predict(self, instances):
        return self.lnr.predict(instances)

    def set_params(self, params: Dict):
        if params['training_instances'] is not None:
            self.set_training_instances(params['training_instances'])
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.lnr = TRIMLearner(self.training_instances,
                               int(0.75 * len(self.training_instances)),
                               verbose=self.verbose)
        self.loss = None
        self.loss_threshold = None
        self.irl_selection = np.full(len(self.training_instances), 1)

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return self.lnr.decision_function(X)
