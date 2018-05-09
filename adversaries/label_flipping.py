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
                 gamma=0.1, num_iterations=10, debug=False):
        Adversary.__init__(self)
        self.learner = learner
        self.cost = cost
        self.total_cost = total_cost
        self.gamma = gamma
        self.num_iterations = 2 * num_iterations
        self.debug = debug

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0 or len(self.cost) != len(instances):
            raise ValueError('Cost data does not match instances.')

        n = len(instances) * 2  # size of new (doubled) input
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

        cost = np.concatenate([np.full(n // 2, 0), np.array(self.cost)])

        # Using alternating minimization. First fix q then minimize epsilon and
        # w. Then, fix epsilon and w and minimize q. The first q will be
        # generated randomly and follow the constraints that the total cost
        # is less than total_cost.
        #
        # Formula: minimize <q, epsilon> + n * self.gamma * (||w||_2 ** 2) -
        # <q, eta>, where <x,y> is the dot product of x and y. See book for
        # constraints.

        ########################################################################
        # Generate initial q by choosing self.total_cost / average_cost values
        # from a uniform distribution and then satisfying constraints
        #
        # average_cost = 0.0
        # for i in self.cost:
        #     average_cost += i
        # average_cost /= len(self.cost)
        # num_to_pick = int(self.total_cost / average_cost)
        #
        # q = np.full(n, 0)
        # tmp = np.random.uniform(0, n // 2, num_to_pick)
        # picked_indices = []
        # for i in tmp:
        #     picked_indices.append(int(i) + n // 2)
        #
        # running_cost = 0.0
        # for i in picked_indices:
        #     if q[i] != 1:  # Only consider if we have not picked it yet
        #         if running_cost + cost[i] <= self.total_cost:# cost constraint
        #             q[i] = 1
        #             running_cost += cost[i]
        #
        # for i in range(n // 2): # Makes sure q[i] + q[i + n // 2] == 1 is True
        #     if q[i] == 0:
        #         if q[i + n // 2] == 0:
        #             q[i] = 1
        #     else:
        #         if q[i + n // 2] == 1:
        #             q[i + n // 2] = 0
        #
        ########################################################################
        # Alternating minimization loop
        # TODO: Use parameters so as to not redefine the problem every time
        # TODO: Use MP to speed up data processing before solving
        # TODO: If all q = np.full(n, 0) works, then delete the above code
        # TODO: Probably do that ^ anyway
        # TODO: Make this more efficient

        q = np.full(n, 0)
        # q = np.concatenate([np.full(n // 2, 0), np.full(n // 2, 1)]) #  GOOD
        # q = np.random.binomial(1, 0.5, n // 2)
        # tmp = []
        # for i in q:
        #     if i == 1:
        #         tmp.append(0)
        #     else:
        #         tmp.append(1)
        # q = np.concatenate([q, np.array(tmp)])
        #
        # for i in q:
        #     assert q[i] + q[i + n // 2] == 1

        old_q, old_epsilon, old_w = np.copy(q), None, None
        flip = True
        for _ in range(self.num_iterations):
            if flip:  # q is fixed, minimize over w and epsilon
                # Setup variables and constants
                epsilon = cvx.Variable(n)
                w = cvx.Variable(instances[0].get_feature_count())
                q = old_q

                # Calculate constants - can possibly ignore this as it only
                # shifts the function - maybe
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
                prob.solve(verbose=self.debug, parallel=True)

                old_epsilon = np.copy(np.array(epsilon.value).flatten())
                old_w = np.copy(np.array(w.value).flatten())

                flip = not flip
            else:  # w and epsilon are fixed, minimize over q
                # Setup variables and constants
                epsilon = old_epsilon
                w = old_w
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
                for i in range(n // 2):
                    constraints.append(q[i] + q[i + n // 2] == 1)
                    cost_for_q += cost[i + n // 2] * q[i + n // 2]
                constraints += [cost_for_q <= self.total_cost]

                prob = cvx.Problem(cvx.Minimize(func), constraints)
                prob.solve(verbose=self.debug, parallel=True)

                q_value = np.array(q.value).flatten()
                old_q = []
                for i in range(n):
                    old_q.append(round(q_value[i]))
                old_q = np.copy(np.array(old_q, dtype=int))
                flip = not flip

        attacked_instances = deepcopy(instances)
        for i in range(n // 2):
            if q[i] == 0:
                label = attacked_instances[i].get_label()
                attacked_instances[i].set_label(-1 * label)
        return attacked_instances


def set_params(self, params: Dict):
    raise NotImplementedError


def get_available_params(self):
    raise NotImplementedError


def set_adversarial_params(self, learner, train_instances):
    raise NotImplementedError
