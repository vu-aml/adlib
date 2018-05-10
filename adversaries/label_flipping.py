# label_flipping.py
# A label flipping implementation

from adversaries.adversary import Adversary
from data_reader.binary_input import Instance
import cvxpy as cvx
import numpy as np
from copy import deepcopy
from typing import List, Dict


class LabelFlipping(Adversary):

    def __init__(self, learner, cost: List[float], total_cost: float,
                 gamma=0.1, num_iterations=10, verbose=False):
        Adversary.__init__(self)
        self.learner = learner
        self.cost = cost
        self.total_cost = total_cost
        self.gamma = gamma
        self.num_iterations = 2 * num_iterations
        self.verbose = verbose
        self.old_q = None
        self.old_epsilon = None
        self.old_w = None

    def attack(self, instances) -> List[Instance]:
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

        q = self._generate_q(half_n)
        self.old_q = np.copy(q)
        flip = True
        for _ in range(self.num_iterations):
            if flip:  # q is fixed, minimize over w and epsilon
                self._minimize_w_epsilon(instances, n, orig_loss,
                                         feature_vectors, labels)
            else:  # w and epsilon are fixed, minimize over q
                self._minimize_q(n, half_n, orig_loss, cost)
            flip = not flip

        ########################################################################

        attacked_instances = deepcopy(instances)
        for i in range(half_n):
            if self.old_q[i] == 0:
                label = attacked_instances[i].get_label()
                attacked_instances[i].set_label(-1 * label)
        return attacked_instances

    def _calculate_constants(self, instances: List[Instance]):
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

    def _generate_q(self, half_n):
        q = np.random.binomial(1, 0.5, half_n)
        tmp = []
        for i in q:
            if i == 1:
                tmp.append(0)
            else:
                tmp.append(1)
        q = np.concatenate([q, np.array(tmp)])

        return q

    def _minimize_w_epsilon(self, instances, n, orig_loss,
                            feature_vectors, labels):
        # Setup variables and constants
        epsilon = cvx.Variable(n)
        w = cvx.Variable(instances[0].get_feature_count())
        q = self.old_q

        # Calculate constants
        cnst = q.dot(orig_loss)

        # Setup CVX problem
        func = self.gamma * n * (cvx.pnorm(w, 2) ** 2) - cnst
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
        prob.solve(verbose=self.verbose, parallel=True)

        self.old_epsilon = np.copy(np.array(epsilon.value).flatten())
        self.old_w = np.copy(np.array(w.value).flatten())

    def _minimize_q(self, n, half_n, orig_loss, cost):
        # Setup variables and constants
        epsilon = self.old_epsilon
        w = self.old_w
        q = cvx.Int(n)

        # Calculate constants - see comment above
        cnst = n * self.gamma * w.dot(w)
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
        prob.solve(verbose=self.verbose, parallel=True)

        q_value = np.array(q.value).flatten()
        self.old_q = []
        for i in range(n):
            self.old_q.append(round(q_value[i]))
        self.old_q = np.copy(np.array(self.old_q, dtype=int))

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
