# outlier_removal_learner.py
# A learner that implements an outlier removal algorithm.
# Matthew Sedam

from adlib.learners.learner import Learner
from adlib.utils.common import get_fvs_and_labels
from typing import Dict
import numpy as np


class OutlierRemovalLearner(Learner):
    """
    A learner that implements an outlier removal algorithm.
    """

    def __init__(self, training_instances, verbose=False):
        Learner.__init__(self)
        self.training_instances = training_instances
        self.verbose = verbose

        self.mean = None
        self.std = None

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        fvs, labels = get_fvs_and_labels(self.training_instances)

        self.mean = np.mean(fvs)
        self.std = np.std(fvs)
        fvs = (fvs - self.mean) / self.std

        iteration = 0
        bad_instances = []
        while iteration == 0 or len(bad_instances) > 0:
            matrix = np.full((fvs.shape[1], fvs.shape[1]), 0.0)
            for fv in fvs:
                fv = fv.reshape((len(fv), 1))
                matrix += fv @ fv.T

    def predict(self, instances):
        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError
