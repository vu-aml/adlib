# alternating_trim_learner.py
# A learner that implements the Alternating TRIM algorithm.
# Matthew Sedam

from adlib.learners import Learner, TRIMLearner
from adlib.utils.common import logistic_loss
from copy import deepcopy
from typing import Dict
import numpy as np


class AlternatingTRIMLearner(Learner):
    """
    A learner that implements the Alternating TRIM algorithm.
    """

    def __init__(self, training_instances, max_iter=10, verbose=False):

        Learner.__init__(self)
        self.set_training_instances(deepcopy(training_instances))
        self.max_iter = max_iter
        self.verbose = verbose

        self.poison_percentage = None
        self.n = None
        self.lnr = TRIMLearner(self.training_instances, 0, verbose=self.verbose)

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        step_size = 1 / len(self.training_instances)

        self.poison_percentage = 0.05
        self.n = int((1 - self.poison_percentage) *
                     len(self.training_instances))
        self.lnr.n = self.n

        poison_percentage_list = []
        loss_list = []
        learner_list = []

        while self.poison_percentage < 0.5:
            self.lnr.train()
            loss = sum(logistic_loss(self.training_instances, self.lnr))

            if self.verbose:
                print('\nPoison Percentage:', self.poison_percentage, '- loss:',
                      loss, '\n')

            poison_percentage_list.append(self.poison_percentage)
            loss_list.append(loss)
            learner_list.append(deepcopy(self.lnr))

            self.poison_percentage += step_size
            self.n = int((1 - self.poison_percentage) *
                         len(self.training_instances))
            self.lnr.n = self.n

        # Find index of smallest change in loss
        loss_list = np.array(loss_list)
        loss_derivative = loss_list[1:] - loss_list[:-1]
        loss_derivative /= step_size
        loss_derivative = np.abs(loss_derivative)
        loss_derivative = list(enumerate(loss_derivative))
        loss_derivative.sort(key=lambda x: x[1])

        index = loss_derivative[0][0]
        index = index if loss_list[index] <= loss_list[index + 1] else index + 1

        self.poison_percentage = poison_percentage_list[index]
        self.n = int((1 - self.poison_percentage) * len(self.training_instances))
        self.lnr = learner_list[index]

    def predict(self, instances):
        return self.lnr.predict(instances)

    def set_params(self, params: Dict):
        if params['training_instances'] is not None:
            self.set_training_instances(deepcopy(params['training_instances']))
        if params['max_iter'] is not None:
            self.max_iter = params['max_iter']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.poison_percentage = None
        self.n = None
        self.lnr = TRIMLearner(self.training_instances, 0, verbose=self.verbose)

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return self.lnr.decision_function(X)
