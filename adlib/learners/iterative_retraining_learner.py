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
        self.loss_threshold = None

    def train(self):
        self.learner.set_training_instances(self.training_instances)
        self.learner.train()
        loss = logistic_loss(self.training_instances, self.learner)

        old_training_instances = []
        while set(old_training_instances) != set(self.training_instances):
            q75, q25 = np.percentile(loss, [75, 25])
            self.loss_threshold = q75 + 1.5 * (q75 - q25)

            old_training_instances = self.training_instances[:]
            instances = []
            for i, inst in enumerate(self.training_instances):
                if loss[i] < self.loss_threshold:
                    instances.append(inst)

            self.training_instances = instances

            if self.verbose:
                print('\nNumber of instances:', len(self.training_instances))
                print('Loss threshold:', self.loss_threshold, '\n')

            self.learner.set_training_instances(self.training_instances)
            self.learner.train()
            loss = logistic_loss(self.training_instances, self.learner)

        self.learner.set_training_instances(self.training_instances)
        self.learner.train()

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

        self.loss_threshold = None

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return self.decision_function(X)
