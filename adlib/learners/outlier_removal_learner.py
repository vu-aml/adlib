# outlier_removal_learner.py
# A learner that implements an outlier removal algorithm.
# Matthew Sedam

from adlib.learners.learner import Learner
from typing import Dict


class OutlierRemovalLearner(Learner):
    """
    A learner that implements an outlier removal algorithm.
    """

    def __init__(self):
        Learner.__init__(self)
        raise NotImplementedError

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        raise NotImplementedError

    def predict(self, instances):
        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError
