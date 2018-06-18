# iterative_retraining_learner.py
# A learner that iteratively retrains and removes outliers based on loss.
# Matthew Sedam

from adlib.learners.learner import learner
from typing import Dict


class IterativeRetrainingLearner(learner):
    """
    A learner that iteratively retrains and removes outliers based on loss.
    """

    def __init__(self):
        learner.__init__(self)
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, instances):
        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError
