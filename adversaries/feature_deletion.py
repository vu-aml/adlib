from data_reader.input import Instance, FeatureVector
from adversaries.adversary import AdversaryStrategy
from typing import List, Dict
from cvxopt import matrix, solvers
import numpy as np
from copy import deepcopy


class Adversary(AdversaryStrategy):

    def __init__(self):
        self.train_instances = None  # type: List[Instance]
        self.num_features = None  # type: int
        self.hinge_loss_multiplier = None  # type: float
        self.max_feature_deletion = None  # type: int
        self.weight_vector = None  # type: np.array

    def change_instances(self, instances: List[Instance]) -> List[Instance]:
        raise NotImplementedError

    def optimize(self):
        """
        Opitimize weight vector using FDROP algorithm
        i.e. formula (6) described in paper
        Returns: optimized weight vector

        """
        num_instances = len(self.train_instances)
        X_list = [ins.feature_vector for ins in self.train_instances]
        y_list = [ins.label for ins in self.train_instances]

        xlen = self.num_features + num_instances * 2 + num_instances * self.num_features
        wi_start = 0
        ti_start = wi_start + self.num_features
        zi_start = ti_start + self.num_features
        v_start = zi_start + self.num_features

        P = np.zeros((xlen, xlen))
        identity = np.multiply(np.identity(self.num_features), 1 / 2)
        P[:self.num_features, :self.num_features] = identity

        weight_terms = [sum(y_list[i] * X_list[i][j] for i in range(num_instances)) * self.weight_vector[j]
                        for j in range(self.num_features)]
        q_list = weight_terms + [1] * num_instances + [0] * (xlen - num_instances - self.num_features)
        q = np.array(q_list)
        np.multiply(q, self.hinge_loss_multiplier)

        G_list = np.array([[0] * xlen])
        for i in range(num_instances):
            lst = [0] * xlen
            lst[ti_start + i] = 1
            lst[zi_start + i] = -self.max_feature_deletion
            lst[v_start + i * self.num_features:v_start + (i + 1) * self.num_features] = [1] * self.num_features
            np.append(G_list, [np.array(lst)])
        for i in range(v_start, xlen):
            lst = [0] * xlen
            lst[i] = 1
            np.append(G_list, [np.array(lst)])
        for i in range(num_instances):
            for j in range(self.num_features):
                lst = [0] * xlen
                lst[wi_start + j] = -y_list[i] * X_list[i][j]
                lst[v_start + i * self.num_features + j] = 1
                lst[zi_start + i] = 1
                np.append(G_list, [np.array(lst)])
        G = G_list[1:]

        h = np.array([0] * len(G))

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

        return sol[:self.num_features]

    def set_params(self, params: Dict):
        if params['hinge_loss_multiplier'] is not None:
            self.hinge_loss_multiplier = params['hinge_loss_multiplier']

        if params['max_feature_deletion'] is not None:
            self.max_feature_deletion = params['max_feature_deletion']

    def get_available_params(self) -> Dict:
        params = {'hinge_loss_multiplier': self.hinge_loss_multiplier,
                  'max_feature_deletion': self.max_feature_deletion}
        return params

    def set_adversarial_params(self, learner, train_instances):
        self.train_instances = train_instances  # type: List[Instance]
        self.num_features = self.train_instances[0].get_feature_vector().get_feature_count()
        self.weight_vector = self.optimize()

    def get_learner_params(self):
        return self.weight_vector

    def predict_instance(self, fv: np.array, a_learner: np.array) -> float:
        return np.dot(self.weight_vector, fv)

