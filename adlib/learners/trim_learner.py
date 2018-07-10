# trim_learner.py
# A learner that implements the TRIM algorithm described in "Manipulating
# Machine Learning - Poisoning Attacks and Countermeasures for Regression
# Learning" found at https://arxiv.org/pdf/1804.00308.pdf.
# Matthew Sedam

from adlib.learners.learner import Learner
from adlib.utils.common import get_fvs_and_labels
from data_reader.binary_input import Instance
from typing import Dict, List
import cvxpy as cvx
import numpy as np


class TRIMLearner(Learner):
    """
    A learner that implements the TRIM algorithm described in the paper
    mentioned above.
    """

    def __init__(self, training_instances: List[Instance], n: int, lda=0.1, verbose=False):
        """
        :param training_instances: the instances on which to train
        :param n: the number of un-poisoned instances in training_instances
                  - the size of the original data set
        :param lda: lambda - for regularization term
        :param verbose: if True, the solver will be in verbose mode
        """

        Learner.__init__(self)
        self.set_training_instances(training_instances)
        self.n = n
        self.lda = lda  # lambda
        self.verbose = verbose

        self.fvs = None
        self.labels = None
        self.tau = None
        self.w = None
        self.b = None

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        self.fvs, self.labels = get_fvs_and_labels(self.training_instances)

        # Create random sample of size self.n
        tmp = np.random.choice(self.fvs.shape[0], self.n, replace=False)
        self.tau = np.full(self.fvs.shape[0], 0)
        for val in tmp:
            self.tau[val] = 1

        # Calculate initial theta
        # Setup variables and constants
        w = cvx.Variable(self.fvs.shape[1])
        b = cvx.Variable()
        tau = cvx.Parameter(self.fvs.shape[0])
        tau.value = self.tau

        # Setup CVX problem
        f_vector = []
        for fv in self.fvs:
            f_vector.append(sum(map(lambda x, y: x * y, w, fv)) + b)

        loss = list(map(lambda x, y: (x - y) ** 2, f_vector, self.labels))
        loss = sum(map(lambda x, y: x * y, tau, loss))
        loss /= self.fvs.shape[0]
        loss += 0.5 * self.lda * (cvx.pnorm(w, 2) ** 2)

        # Solve problem
        prob = cvx.Problem(cvx.Minimize(loss), [])
        prob.solve(solver=cvx.ECOS, verbose=self.verbose, parallel=True, ignore_dcp=True)
        self.w, self.b = np.array(w.value).flatten(), b.value

        old_loss = -1
        loss = 0
        iteration = 0
        while loss != old_loss:
            if self.verbose:
                print('\nTRIM Iteration:', iteration, '- current loss:', loss,
                      '\n')

            # Calculate minimal set
            loss_vector = self.fvs.dot(self.w) + self.b
            loss_vector -= self.labels
            loss_vector = loss_vector ** 2

            # Sort based on loss and take self.n instances
            loss_tuples = list(enumerate(loss_vector))
            loss_tuples.sort(key=lambda tup: tup[1])
            loss_tuples = loss_tuples[:self.n]

            self.tau = np.full(len(self.tau), 0)
            for index, _ in loss_tuples:
                self.tau[index] = 1

            # Minimize loss
            tau.value = self.tau
            prob.solve(solver=cvx.ECOS, verbose=self.verbose, parallel=True,
                       warm_start=True, ignore_dcp=True)
            self.w, self.b = np.array(w.value).flatten(), b.value

            old_loss = loss
            loss = self._calc_loss()
            iteration += 1

    def _calc_loss(self):
        """
        Calculates the loss function as specified in the paper.
        :return: the loss
        """

        loss = 0.5 * self.lda * (np.linalg.norm(self.w) ** 2)
        tmp = sum(map(lambda x, y: (x - y) ** 2, self.fvs.dot(self.w) + self.b, self.labels))
        loss += tmp / self.fvs.shape[0]

        return loss

    def predict(self, instances):
        """
        Predict classification labels for the set of instances.
        :param instances: list of Instance objects
        :return: label classifications (List(int))
        """

        if self.w is None or self.b is None:
            raise ValueError('Must train learner before prediction.')

        fvs, _ = get_fvs_and_labels(instances)

        labels = fvs.dot(self.w) + self.b
        labels = list(map(lambda x: 1 if x >= 0 else -1, labels))

        return labels

    def set_params(self, params: Dict):
        """
        Sets parameters for the learner.
        :param params: parameters
        """

        if params['training_instances'] is not None:
            self.set_training_instances(params['training_instances'])
        if params['n'] is not None:
            self.n = params['n']
        if params['lda'] is not None:
            self.lda = params['lda']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.fvs = None
        self.labels = None
        self.tau = None
        self.w = None
        self.b = None

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        if self.w is None or self.b is None:
            raise ValueError('Must train learner before decision_function.')

        return X.dot(self.w) + self.b
