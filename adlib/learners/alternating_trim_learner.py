# alternating_trim_learner.py
# A learner that implements the Alternating TRIM algorithm.
# Matthew Sedam

from adlib.learners.learner import Learner
from adlib.utils.common import get_fvs_and_labels, logistic_loss
from copy import deepcopy
from typing import Dict
import cvxpy as cvx
import numpy as np


class AlternatingTRIMLearner(Learner):
    """
    A learner that implements the Alternating TRIM algorithm.
    """

    def __init__(self, training_instances, poison_percentage, max_iter=50,
                 verbose=False):

        Learner.__init__(self)
        self.set_training_instances(deepcopy(training_instances))
        self.poison_percentage = poison_percentage
        self.n = int((1 - poison_percentage) * len(self.training_instances))
        self.max_iter = max_iter
        self.verbose = verbose
        self.theta = None
        self.b = None

    def train(self):
        fvs, labels = get_fvs_and_labels(self.training_instances)
        tau = self._generate_tau()
        old_tau = np.full(len(tau), 0)
        tau_dist = int(np.linalg.norm(tau - old_tau) ** 2)
        iteration = 0

        while tau_dist != 0 and iteration < self.max_iter:
            print('Iteration: ', iteration, ' - tau_dist: ', tau_dist, sep='')

            # Setup variables
            theta = cvx.Variable(fvs.shape[1])
            b = cvx.Variable()

            # Setup CVXPY problem
            f_vector = []
            for vector in fvs:
                f_vector.append(sum(map(lambda x, y: x * y, vector, theta)) + b)

            tmp = []
            for i, val in enumerate(tau):
                if val == 1:
                    tmp.append(cvx.logistic(-1 * labels[i] * f_vector[i]))

            # Solve the minimization problem
            func = sum(tmp)
            problem = cvx.Problem(cvx.Minimize(func), [])
            problem.solve(solver=cvx.ECOS, verbose=self.verbose, parallel=True)

            self.theta = np.array(theta.value).flatten()
            self.b = b.value

            ####################################################################

            loss = logistic_loss(self.training_instances, self)
            loss_sort_list = list(enumerate(loss))
            loss_sort_list.sort(key=lambda x: x[1])
            old_tau = deepcopy(tau)

            for i, val in enumerate(loss_sort_list):
                tau[val[0]] = 1 if i < self.n else 0

            tau_dist = int(np.linalg.norm(tau - old_tau) ** 2)
            iteration += 1

        print('Iteration: FINAL - tau_dist: ', tau_dist, sep='')

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
            for i, val in enumerate(tau):
                if total > self.n:
                    if val == 1 and np.random.binomial(1, 0.5) == 1:
                        tau[i] = 0
                        break
                else:
                    if val == 0 and np.random.binomial(1, 0.5) == 1:
                        tau[i] = 1
                        break
            total = sum(tau)

        return tau

    def predict(self, instances):
        fvs, _ = get_fvs_and_labels(instances)
        decision_vals = self.decision_function(fvs)
        return list(map(lambda x: 1 if x >= 0 else -1, decision_vals))

    def set_params(self, params: Dict):
        if params['training_instances'] is not None:
            self.set_training_instances(params['training_instances'])
        if params['poison_percentage'] is not None:
            self.poison_percentage = params['poison_percentage']
        if params['max_iter'] is not None:
            self.max_iter = params['max_iter']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return X.dot(self.theta) + self.b
