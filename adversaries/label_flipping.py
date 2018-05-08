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
                 gamma=0.05):
        Adversary.__init__(self)
        self.learner = learner
        self.cost = cost
        self.total_cost = total_cost
        self.gamma = gamma

    def attack(self, instances) -> List[Instance]:
        if len(self.cost) != len(instances):
            raise ValueError('Cost data does not match instances.')

        n = len(instances) * 2
        pred_labels = self.learner.predict(instances)
        orig_loss = []
        for i in range(len(pred_labels)):
            orig_loss.append((pred_labels[i] - instances[i].get_label()) ** 2)
        orig_loss = np.array(orig_loss + orig_loss)

        # Setup CVX
        q = cvx.Int(n)
        w = cvx.Variable(instances[0].get_feature_count())
        epsilon = cvx.Variable(n)

        func = q.T * (epsilon - orig_loss) + n * self.gamma * (cvx.norm(w) ** 2)

        feature_vectors = []
        labels = []
        for inst in instances:
            feature_vectors.append(inst.get_feature_vector())
            labels.append(inst.get_label())
        feature_vectors = np.array(feature_vectors + feature_vectors)
        labels = np.array(labels + labels)

        cost = np.concatenate([np.full(int(n / 2), 0), np.array(self.cost)])

        constraints = [cost.T * q <= self.total_cost]
        for i in range(n):
            constraints.append(1 - labels[i] * w.T * feature_vectors[i]
                               <= epsilon[i])
            constraints.append(0 <= epsilon[i])
            constraints.append(0 <= q[i])
            constraints.append(q[i] <= 1)
            if i < int(n / 2):
                constraints.append(q[i] + q[i + int(n / 2)] == 1)

        prob = cvx.Problem(cvx.Minimize(func), constraints)
        result = prob.solve()
        print(result)  # ####################

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
