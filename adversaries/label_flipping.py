# label_flipping.py
# A label flipping implementation

from adversaries.adversary import Adversary
from data_reader.binary_input import Instance
import cvxpy as cvx
import numpy as np
from typing import List, Dict


class FeatureVectorWrapper(np.ndarray):
    """
    Wraps a feature vector to look like an np.ndarray. This is needed as only
    np.ndarray and Python lists can be used for CVX, with np.ndarray being
    preferred. This wraps a dummy array and encapsulates the feature_vector
    so as to provide an np.ndarray interface to the efficient representation of
    a feature_vector. Using the [] operator directly on a feature_vector does
    not return 0/1 for the feature, but directly indexes into the underlying
    representation. The base code and inspiration for this came from the
    official numpy documentation.
    """

    def __new__(cls, feature_vector, info=None):
        # input_array is an np_array, use a dummy object
        obj = np.asarray(np.full(1, 0)).view(cls)
        obj.feature_vector = feature_vector
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.feature_vector = getattr(obj, 'feature_vector', None)

    def __getitem__(self, key):
        return self.feature_vector.get_feature(key)


class LabelFlipping(Adversary):

    def __init__(self, learner, cost: List[float], total_cost: float,
                 gamma=0.05, num_iterations=10):
        Adversary.__init__(self)
        self.learner = learner
        self.cost = cost
        self.total_cost = total_cost
        self.gamma = gamma
        self.num_iterations = num_iterations

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0 or len(self.cost) != len(instances):
            raise ValueError('Cost data does not match instances.')

        n = len(instances) * 2  # size of new (doubled) input
        pred_labels = self.learner.predict(instances)
        orig_loss = []
        for i in range(len(pred_labels)):
            orig_loss.append((pred_labels[i] - instances[i].get_label()) ** 2)
        orig_loss = np.array(orig_loss + orig_loss)  # eta in formula

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

        cost = np.concatenate([np.full(int(n / 2), 0), np.array(self.cost)])
        average_cost = 0.0
        for i in self.cost:
            average_cost += i
        average_cost /= len(self.cost)
        num_to_pick = int(self.total_cost / average_cost)

        q = np.full(n, 0)
        tmp = np.random.uniform(0, n, num_to_pick)
        picked_indices = []
        for i in tmp:
            picked_indices.append(int(i))

        running_cost = 0.0
        for i in picked_indices:
            if q[i] != 1:  # Only consider if we have not picked it yet
                if running_cost + cost[i] <= self.total_cost:  # cost constraint
                    q[i] = 1
                    running_cost += cost[i]

        for i in range(n // 2):  # Makes sure q[i] + q[i + n // 2] == 1 is True
            if q[i] == 0:
                if q[i + n // 2] == 0:
                    q[i] = 1
            else:
                if q[i + n // 2] == 1:
                    q[i + n // 2] = 0

        ########################################################################
        # Alternating minimization loop

        old_q, old_epsilon, old_w = q, None, None
        flip = True
        for i in range(self.num_iterations):
            if flip:  # q is fixed, minimize over w and epsilon
                # Setup variables
                w = cvx.Variable(instances[0].get_feature_count())
                epsilon = cvx.Variable(n)

                # Calculate constants - can possibly ignore this as it only
                # shifts the function - maybe
                cnst = q.dot(orig_loss)

                # Setup CVX problem
                func = self.gamma * n * (cvx.pnorm(w, 2) ** 2) - cnst
                for i in range(n):
                    func += q[i] * epsilon[i]

                feature_vectors = []
                labels = []
                for inst in instances:
                    feature_vectors.append(
                        FeatureVectorWrapper(inst.get_feature_vector()))
                    labels.append(inst.get_label())
                feature_vectors = np.array(feature_vectors + feature_vectors)
                labels = np.array(labels + labels)

                constraints = []
                for i in range(n):
                    constraints.append(1 - labels[i] * w.T * feature_vectors[i]
                                       <= epsilon[i])
                    constraints.append(0 <= epsilon[i])

                prob = cvx.Problem(cvx.Minimize(func), constraints)
                prob.solve()

                old_epsilon = np.array(epsilon.value).flatten()
                old_w = np.array(w.value).flatten()
            # else:  # w and epsilon are fixed, minimize over q
            #     q = cvx.Int(n)


def set_params(self, params: Dict):
    raise NotImplementedError


def get_available_params(self):
    raise NotImplementedError


def set_adversarial_params(self, learner, train_instances):
    raise NotImplementedError
