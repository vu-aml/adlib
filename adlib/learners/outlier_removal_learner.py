# outlier_removal_learner.py
# A learner that implements an outlier removal algorithm.
# Matthew Sedam

from adlib.learners.learner import Learner
from adlib.utils.common import get_fvs_and_labels
from typing import Dict
import math
import numpy as np


class OutlierRemovalLearner(Learner):
    """
    A learner that implements an outlier removal algorithm.
    """

    def __init__(self, training_instances, verbose=False):
        Learner.__init__(self)
        self.training_instances = training_instances
        self.verbose = verbose

        self.w = None
        self.mean = None
        self.std = None

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        fvs, labels = get_fvs_and_labels(self.training_instances)
        fvs, labels = self._remove_outliers(fvs, labels)

        self.w = np.full(fvs.shape[1], 0.0)
        for i, fv in enumerate(fvs):
            self.w += labels[i] * fv
        self.w /= fvs.shape[0]

    def _remove_outliers(self, fvs, labels):
        """
        Removes outliers
        :param fvs: the feature vectors - np.ndarray
        :param labels: the labels
        :return: feature vectors and labels
        """

        cutoff = 10 * math.log(len(self.training_instances)) / fvs.shape[1]

        if self.verbose:
            print('Cutoff:', cutoff)

        self.mean = np.mean(fvs)
        self.std = np.std(fvs)
        fvs = (fvs - self.mean) / self.std

        iteration = 0
        old_number_of_instances = -1
        while old_number_of_instances != len(labels):
            if self.verbose:
                print('Iteration:', iteration, '- num_instances:', len(labels))

            matrix = np.full((fvs.shape[1], fvs.shape[1]), 0.0)
            for fv in fvs:
                fv = fv.reshape((len(fv), 1))
                matrix += fv @ fv.T

            eigen_vals, eigen_vectors = np.linalg.eig(matrix)

            tmp = []
            for val in eigen_vals:
                tmp.append(np.linalg.norm(val))
            eigen_vals = tmp

            # Find eigenvector with largest eigenvalue
            eigen_vals = list(enumerate(eigen_vals))
            eigen_vals.sort(key=lambda x: x[1])
            largest_index = eigen_vals[-1][0]
            v = eigen_vectors[largest_index]

            new_fvs = []
            new_labels = []
            for i, fv in enumerate(fvs):
                value = v.dot(fv) ** 2
                if value < cutoff:
                    new_fvs.append(fv)
                    new_labels.append(labels[i])

            old_number_of_instances = len(labels)
            fvs = np.array(new_fvs)
            labels = np.array(new_labels)

            iteration += 1

        return fvs, labels

    def predict(self, instances):
        fvs, _ = get_fvs_and_labels(instances)
        return list(map(lambda x: 1 if x > 0 else -1, self.decision_function(fvs)))

    def set_params(self, params: Dict):
        if params['training_instances'] is not None:
            self.training_instances = params['training_instances']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.w = None
        self.mean = None
        self.std = None

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        X = (X - self.mean) / self.std
        return X.dot(self.w)
