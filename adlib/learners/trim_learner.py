# trim_learner.py
# A learner that implements the TRIM algorithm described in "Manipulating
# Machine Learning- Poisoning Attacks and Countermeasures for Regression
# Learning" found at https://arxiv.org/pdf/1804.00308.pdf.
# Matthew Sedam

from adlib.learners.learner import learner
from typing import Dict


class TRIM_Learner(learner):
    """
    A learner that implements the TRIM algorithm described in the paper
    mentioned above.
    """

    def __init__(self):
        learner.__init__(self)
        raise NotImplementedError

    def train(self):
        """
        Train on the set of training instances.
        """

        raise NotImplementedError

    def predict(self, instances):
        """
        Predict classification labels for the set of instances.
        :param instances: list of Instance objects
        :return: label classifications (List(int))
        """

        raise NotImplementedError

    def set_params(self, params: Dict):
        """
        Sets parameters for the learner.
        :param params: parameters
        """

        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError
