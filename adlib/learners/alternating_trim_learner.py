# alternating_trim_learner.py
# A learner that implements the Alternating TRIM algorithm.
# Matthew Sedam

from adlib.learners.learner import learner
from adlib.utils.common import get_fvs_and_labels, logistic_loss
from copy import deepcopy
from typing import Dict
import numpy as np


class AlternatingTRIMLearner(learner):
    """
    A learner that implements the Alternating TRIM algorithm.
    """

    def __init__(self, training_instances, poison_percentage, verbose):
        learner.__init__(self)
        self.training_instances = deepcopy(training_instances)
        self.poison_percentage = poison_percentage
        self.n = (1 - poison_percentage) * len(self.training_instances)
        self.verbose = verbose

    def train(self):
        fvs, labels = get_fvs_and_labels()
        tau = self._generate_tau()

    def _generate_tau(self):
        """
        Generates a random tau, where tau[i] in {0, 1} for i in range(len(tau))
        and sum(tau) = self.n
        :return: tau
        """

        tau = np.random.binomial(1, 1 - self.poison_percentage,
                                 len(self.training_instances))

        total = sum(tau)
        while total != self.n:
            if total > self.n:
                for i, val in enumerate(tau):
                    if val == 1 and np.random.binomial(1, 0.5) == 1:
                        tau[i] = 0
                        break
            else:
                for i, val in enumerate(tau):
                    if val == 0 and np.random.binomial(1, 0.5) == 1:
                        tau[i] = 1
                        break

        return tau

    def predict(self, instances):
        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError
