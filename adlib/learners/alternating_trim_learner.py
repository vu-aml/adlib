# alternating_trim_learner.py
# A learner that implements the Alternating TRIM algorithm.
# Matthew Sedam

from adlib.learners.learner import learner
from typing import Dict


class AlternatingTRIMLearner(learner):
    """
    A learner that implements the Alternating TRIM algorithm.
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
