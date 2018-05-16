# label_flipping.py
# A label flipping implementation
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
import cvxpy as cvx
import numpy as np
from copy import deepcopy
from progress.bar import Bar
from typing import List, Dict


class LabelFlipping(Adversary):
    """
    This performs a label flipping attack on a set of Instances by maximizing
    the utility of the attacker, while minimizing risk and maximizing the
    amount that the trained model will differ from the original (true) model.
    """

    def __init__(self, learner, cost: List[float], total_cost: float,
                 gamma=0.1, num_iterations=10, verbose=False):
        """
        :param learner: the previously-trained SVM learner
        :param cost: the cost vector, has length of size of instances
        :param total_cost: the total cost for the attack
        :param gamma: the gamma rate, default 0.1
        :param num_iterations: the number of iterations of the alternating
                               minimization loop, default 10
        :param verbose: if True, then the solver will be set to verbose mode,
                        default False
        """

        Adversary.__init__(self)
        self.learner = learner
        self.cost = cost
        self.total_cost = total_cost
        self.gamma = gamma
        self.num_iterations = 2 * num_iterations
        self.verbose = verbose
        self._old_q = None
        self._old_epsilon = None
        self._old_w = None

    def attack(self, instances) -> List[Instance]:
        """
        Takes instances and performs a label flipping attack.
        :param instances: The list of Instances
        :return: The attacked instances with labels flipped if deemed "good" by
                 the solver.
        """

        if len(instances) == 0 or len(self.cost) != len(instances):
            raise ValueError('Cost data does not match instances.')

        print('Start label flipping attack.\n')

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

        q = np.full(n, 0)
        self._old_q = np.copy(q)
        flip = True
        if not self.verbose:
            bar = Bar('Processing', max=self.num_iterations + 1,
                      suffix='%(percent)d%%')
            bar.next()
        for _ in range(self.num_iterations):
            if flip:  # q is fixed, minimize over w and epsilon
                self._minimize_w_epsilon(instances, n, orig_loss,
                                         feature_vectors, labels)
            else:  # w and epsilon are fixed, minimize over q
                self._minimize_q(n, half_n, orig_loss, cost)
            flip = not flip
            if not self.verbose:
                bar.next()
        if not self.verbose:
            bar.finish()

        ########################################################################

        attacked_instances = deepcopy(instances)
        for i in range(half_n):
            if self._old_q[i] == 0:
                label = attacked_instances[i].get_label()
                attacked_instances[i].set_label(-1 * label)

        print('End label flipping attack.\n')

        return attacked_instances

    def _calculate_constants(self, instances: List[Instance]):
        """
        Calculates constants needed for the alternating minimization loop.
        :param instances: the list of Instances
        :return: the constants
        """

        half_n = len(instances)
        n = half_n * 2  # size of new (doubled) input
        pred_labels = self.learner.predict(instances)
        orig_loss = []
        for i in range(len(pred_labels)):
            orig_loss.append((pred_labels[i] - instances[i].get_label()) ** 2)
        orig_loss = np.array(orig_loss + orig_loss)  # eta in formula

        feature_vectors = []
        labels = []
        labels_flipped = []
        for inst in instances:
            feature_vectors.append(inst.get_feature_vector())
            labels.append(inst.get_label())
            labels_flipped.append(-inst.get_label())
        feature_vectors = np.array(feature_vectors + feature_vectors)
        labels = np.array(labels + labels_flipped)

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
        q = self._old_q

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
                if feature_vectors[i].get_feature(j) == 1:
                    tmp += w[j]
            constraints.append(1 - labels[i] * tmp <= epsilon)
            constraints.append(0 <= epsilon[i])

        prob = cvx.Problem(cvx.Minimize(func), constraints)
        prob.solve(verbose=self.verbose, parallel=True,
                   abstol_inacc=1e-2, reltol_inacc=1e-2, feastol_inacc=1e-2)

        self._old_epsilon = np.copy(np.array(epsilon.value).flatten())
        self._old_w = np.copy(np.array(w.value).flatten())

    def _minimize_q(self, n, half_n, orig_loss, cost):
        """
        Minimize over q while keeping epsilon and w constant.
        :param n: the number of instances
        :param half_n: half of n
        :param orig_loss: the original loss calculations
        :param cost: cost: the cost vector, has length of size of instances
        """

        # Setup variables and constants
        epsilon = self._old_epsilon
        w = self._old_w
        q = cvx.Int(n)

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
        constraints += [cost_for_q <= self.total_cost]

        prob = cvx.Problem(cvx.Minimize(func), constraints)
        prob.solve(verbose=self.verbose, parallel=True, solver=cvx.ECOS_BB)

        q_value = np.array(q.value).flatten()
        self._old_q = []
        for i in range(n):
            self._old_q.append(round(q_value[i]))
        self._old_q = np.copy(np.array(self._old_q, dtype=int))

    def set_params(self, params: Dict):
        if params['learner'] is not None:
            self.learner = params['learner']
        if params['cost'] is not None:
            self.cost = params['cost']
        if params['total_cost'] is not None:
            self.total_cost = params['total_cost']
        if params['gamma'] is not None:
            self.gamma = params['gamma']
        if params['num_iterations'] is not None:
            self.num_iterations = params['num_iterations']
        if params['verbose'] is not None:
            self.verbose = params['verbose']
        self._old_q = None
        self._old_epsilon = None
        self._old_w = None

    def get_available_params(self):
        params = {'learner': self.learner,
                  'cost': self.cost,
                  'total_cost': self.total_cost,
                  'gamma': self.gamma,
                  'num_iterations': self.num_iterations,
                  'verbose': self.verbose}
        return params

    def set_adversarial_params(self, learner, train_instances):
        self.learner = learner
