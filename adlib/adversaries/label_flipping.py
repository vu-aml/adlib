# label_flipping.py
# A label flipping implementation
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from adlib.utils.common import get_fvs_and_labels, logistic_loss
from data_reader.binary_input import Instance
import cvxpy as cvx
import numpy as np
from copy import deepcopy
from typing import List, Dict


class LabelFlipping(Adversary):
    """
    This performs a label flipping attack on a set of Instances by maximizing
    the utility of the attacker, while minimizing risk and maximizing the
    amount that the trained model will differ from the original (true) model.
    """

    def __init__(self, learner, cost: List[float], total_cost: float,
                 gamma=0.1, alpha=5e-7, verbose=False):
        """
        :param learner: the previously-trained SVM learner
        :param cost: the cost vector, has length of size of instances
        :param total_cost: the total cost for the attack
        :param gamma: the gamma rate, default 0.1
        :param alpha: the convergence level
        :param verbose: if True, then the solver will be set to verbose mode,
                        default False
        """

        Adversary.__init__(self)
        self.learner = learner
        self.cost = cost
        self.total_cost = total_cost
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose
        self.q = None
        self.epsilon = None
        self.w = None

    def attack(self, instances) -> List[Instance]:
        """
        Takes instances and performs a label flipping attack.
        :param instances: The list of Instances
        :return: The attacked instances with labels flipped if deemed "good" by
                 the solver.
        """

        if len(instances) == 0 or len(self.cost) != len(instances):
            raise ValueError('Cost data does not match instances.')

        (half_n,
         n,
         orig_loss,
         feature_vectors,
         labels,
         cost) = self._calculate_constants(instances)

        ########################################################################
        # Using alternating minimization. First fix q then minimize epsilon and
        # w. Then, fix epsilon and w and minimize q. The first q will be
        # generated randomly and follow the constraints that the total cost
        # is less than total_cost.
        #
        # Formula: minimize <q, epsilon> + n * self.gamma * (||w||_2 ** 2) -
        # <q, eta>, where <x,y> is the dot product of x and y. See book for
        # constraints.
        ########################################################################
        # Alternating minimization loop

        q = np.random.rand(half_n)
        q_add_inv = 1 - q
        q = np.concatenate([q, q_add_inv])

        self.q = deepcopy(q)
        q_dist = 0
        iteration = 0

        while iteration == 0 or q_dist > self.alpha:
            print('\nIteration:', iteration, '- q_dist:', q_dist, '- q:')
            print(self.q, '\n')

            old_q = deepcopy(self.q)
            self._minimize_w_epsilon(instances, n, orig_loss,
                                     feature_vectors, labels)
            self._minimize_q(n, half_n, orig_loss, cost)
            q_dist = np.linalg.norm(self.q - old_q)
            iteration += 1

        print('\nIteration: FINAL - q_dist:', q_dist)

        attacked_instances = deepcopy(instances)
        for i in range(half_n):
            if self.q[i] <= 0.5:
                label = attacked_instances[i].get_label()
                attacked_instances[i].set_label(-1 * label)

        return attacked_instances

    def _calculate_constants(self, instances: List[Instance]):
        """
        Calculates constants needed for the alternating minimization loop.
        :param instances: the list of Instances
        :return: the constants
        """

        half_n = len(instances)
        n = half_n * 2  # size of new (doubled) input

        feature_vectors = []
        labels = []
        labels_flipped = []
        for inst in instances:
            feature_vectors.append(inst.get_feature_vector())
            labels.append(inst.get_label())
            labels_flipped.append(-1 * inst.get_label())
        feature_vectors = np.array(feature_vectors + feature_vectors)
        labels = np.array(labels + labels_flipped)

        fvs, _ = get_fvs_and_labels(instances)
        orig_loss = logistic_loss(fvs, self.learner,
                                  np.array(labels[:(len(labels) // 2)]))
        orig_loss = np.concatenate([orig_loss, orig_loss])

        cost = np.concatenate([np.full(half_n, 0), np.array(self.cost)])

        return half_n, n, orig_loss, feature_vectors, labels, cost

    def _minimize_w_epsilon(self, instances, n, orig_loss,
                            feature_vectors, labels):
        """
        Minimizes over w and epsilon while keeping q constant. First iteration
        of the alternating minimization loop.
        :param instances: the list of Instances
        :param n: the number of instances
        :param orig_loss: the original loss calculations
        :param feature_vectors: the list of feature vectors
        :param labels: the list of labels
        """

        # Setup variables and constants
        epsilon = cvx.Variable(n)
        w = cvx.Variable(instances[0].get_feature_count())
        q = self.q

        # Calculate constants
        cnst = q.dot(orig_loss)

        # Setup CVX problem
        func = self.gamma * (cvx.pnorm(w, 2) ** 2) - cnst
        for i in range(n):
            func += q[i] * epsilon[i]

        constraints = []
        for i in range(n):
            tmp = 0.0
            for j in range(instances[0].get_feature_count()):
                if feature_vectors[i].get_feature(j) > 0:
                    tmp += w[j] * feature_vectors[i].get_feature(j)
            constraints.append(1 - labels[i] * tmp <= epsilon)
            constraints.append(0 <= epsilon[i])

        prob = cvx.Problem(cvx.Minimize(func), constraints)
        prob.solve(solver=cvx.ECOS, verbose=self.verbose, parallel=True)

        self.epsilon = np.copy(np.array(epsilon.value).flatten())
        self.w = np.copy(np.array(w.value).flatten())

    def _minimize_q(self, n, half_n, orig_loss, cost):
        """
        Minimize over q while keeping epsilon and w constant.
        :param n: the number of instances
        :param half_n: half of n
        :param orig_loss: the original loss calculations
        :param cost: the cost vector, has length of size of instances
        """

        # Setup variables and constants
        epsilon = self.epsilon
        w = self.w
        q = cvx.Variable(n)

        # Calculate constants - see comment above
        cnst = self.gamma * w.dot(w)
        epsilon_diff_eta = epsilon - orig_loss

        # Setup CVX problem
        func = cnst
        for i in range(n):
            func += q[i] * epsilon_diff_eta[i]

        constraints = [0 <= q, q <= 1]
        cost_for_q = 0.0
        for i in range(half_n):
            constraints.append(q[i] + q[i + half_n] == 1)
            cost_for_q += cost[i + half_n] * q[i + half_n]
        constraints.append(cost_for_q <= self.total_cost)

        prob = cvx.Problem(cvx.Minimize(func), constraints)
        prob.solve(solver=cvx.ECOS, verbose=self.verbose, parallel=True)

        q_value = np.array(q.value).flatten()
        self.q = q_value

    def set_params(self, params: Dict):
        if params['learner'] is not None:
            self.learner = params['learner']
        if params['cost'] is not None:
            self.cost = params['cost']
        if params['total_cost'] is not None:
            self.total_cost = params['total_cost']
        if params['gamma'] is not None:
            self.gamma = params['gamma']
        if params['alpha'] is not None:
            self.alpha = params['alpha']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.q = None
        self.epsilon = None
        self.w = None

    def get_available_params(self):
        params = {'learner': self.learner,
                  'cost': self.cost,
                  'total_cost': self.total_cost,
                  'gamma': self.gamma,
                  'alpha': self.alpha,
                  'verbose': self.verbose}
        return params

    def set_adversarial_params(self, learner, train_instances):
        self.learner = learner
