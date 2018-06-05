# trim_learner.py
# A learner that implements the TRIM algorithm described in "Manipulating
# Machine Learning - Poisoning Attacks and Countermeasures for Regression
# Learning" found at https://arxiv.org/pdf/1804.00308.pdf.
# Matthew Sedam

from data_reader.binary_input import Instance
from adlib.learners.learner import learner
from typing import Dict, List
import cvxpy as cvx
import numpy as np


class TRIM_Learner(learner):
    """
    A learner that implements the TRIM algorithm described in the paper
    mentioned above.
    """

    def __init__(self, training_instances: List[Instance], n: int, lda=0.1,
                 verbose=False):
        """
        :param training_instances: the instances on which to train
        :param n: the number of unpoisoned instances in training_instances - the
                  size of the original dataset
        :param lda: lambda - for regularization term
        :param verbose: if True, the solver will be in verbose mode
        """

        learner.__init__(self)
        self.training_instances = training_instances
        self.n = n
        self.lda = lda  # lambda
        self.verbose = verbose
        self.num_features = self.training_instances[0].get_feature_count()
        self.w = None
        self.b = None

    def train(self):
        """
        Train on the set of training instances.
        """

        # Create random sample of size self.n
        inst_set = []
        while len(inst_set) < self.n:
            for inst in self.training_instances:
                if np.random.binomial(1, 0.5) == 1 and len(inst_set) < self.n:
                    inst_set.append(inst)

                if len(inst_set) == self.n:
                    break

        fvs, labels = TRIM_Learner.get_fv_matrix_and_labels(inst_set)

        # Calculate initial theta
        w, b = self._minimize_loss(fvs, labels)

        old_loss = -1
        loss = 0
        while loss != old_loss:
            # Calculate minimal set
            loss_vector = fvs.dot(w) + b
            loss_vector -= labels
            loss_vector = list(map(lambda x: x ** 2, loss_vector))

            loss_tuples = []
            for i in range(len(loss_vector)):
                loss_tuples.append((loss_vector[i], inst_set[i]))
            loss_tuples.sort()

            inst_set = list(map(lambda tup: tup[1], loss_tuples[:self.n]))

            # Minimize loss
            fvs, labels = TRIM_Learner.get_fv_matrix_and_labels(inst_set)
            w, b = self._minimize_loss(fvs, labels)

            old_loss = loss
            loss = self._calc_loss(inst_set)

        self.w = w
        self.b = b

    def _minimize_loss(self, fvs, labels):
        """
        Use CVXPY to minimize the loss function.
        :param fvs: the feature vectors (np.ndarray)
        :param labels: the list of labels (np.ndarray or List[int])
        :return: w (np.ndarray) and b (float)
        """

        # Setup variables and constants
        w = cvx.Variable(fvs.shape[1])
        b = cvx.Variable()

        # Setup CVX problem
        f_vector = []
        for arr in fvs:
            f_vector.append(sum(map(lambda x, y: x * y, w, arr)) + b)

        loss = sum(map(lambda x, y: (x - y) ** 2, f_vector, labels))
        loss /= fvs.shape[0]
        loss += 0.5 * self.lda * (cvx.pnorm(w, 2) ** 2)

        # Solve problem
        prob = cvx.Problem(cvx.Minimize(loss), [])
        prob.solve(verbose=self.verbose, parallel=True)

        return np.array(w.value), b.value

    def _calc_loss(self, fvs, labels, w, b):
        """
        Calculates the loss function as specified in the paper
        :param inst_set: the set of Instances
        :return: the loss
        """

        loss = 0.5 * self.lda * (np.linalg.norm(w) ** 2)
        tmp = sum(map(lambda x, y: (x - y) ** 2, fvs.dot(w) + b, labels))
        loss += tmp / fvs.shape[0]

        return loss

    def predict(self, instances):
        """
        Predict classification labels for the set of instances.
        :param instances: list of Instance objects
        :return: label classifications (List(int))
        """

        if not self.w or not self.b:
            raise ValueError('Must train learner before prediction.')

        fvs, _ = TRIM_Learner.get_fv_matrix_and_labels(instances)

        labels = fvs.dot(self.w) + self.b
        labels = list(map(lambda x: 1 if x >= 0 else -1, labels))

        return labels

    def set_params(self, params: Dict):
        """
        Sets parameters for the learner.
        :param params: parameters
        """

        if params['training_instances'] is not None:
            self.training_instances = params['training_instances']
        if params['n'] is not None:
            self.n = params['n']
        if params['lda'] is not None:
            self.lda = params['lda']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.num_features = self.training_instances[0].get_feature_count()
        self.w = None
        self.b = None

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return X.dot(self.w) + self.b

    @staticmethod
    def get_feature_vector_array(inst: Instance):
        """
        Turns the feature vector into an np.ndarray
        :param inst: the Instance
        :return: the feature vector (np.ndarray)
        """

        fv = inst.get_feature_vector()
        tmp = []
        for j in range(inst.get_feature_count()):
            if fv.get_feature(j) == 1:
                tmp.append(1)
            else:
                tmp.append(0)
        return np.array(tmp)

    @staticmethod
    def get_fv_matrix_and_labels(instances: List[Instance]):
        """
        Calculate feature vector matrix and label array
        :param instances: the list of Instances
        :return: a tuple of the feature vector matrix and labels
                 (np.ndarray, np.ndarray)
        """

        fvs = []
        labels = []
        for inst in instances:
            fvs.append(TRIM_Learner.get_feature_vector_array(inst))
            labels.append(inst.get_label())

        return np.array(fvs), np.array(labels)
