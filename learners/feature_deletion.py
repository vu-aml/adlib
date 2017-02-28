from learners.learner import RobustLearner

from data_reader.input import Instance
from typing import List, Dict
import numpy as np
from cvxopt import matrix, solvers





class FeatureDeletion(RobustLearner):

    def __init__(self, params=None, training_instances=None):
        RobustLearner.__init__(self)
        self.weight_vector = None
        self.num_features = None  # type: int
        self.hinge_loss_multiplier = 0.5  # type: float
        self.max_feature_deletion = 30  # type: int

        self.weight_vector = None  # type: np.array
        self.bias = 0  # type: int

        self.set_params(params)
        self.set_training_instances(training_instances)

    def set_params(self, params: Dict):
        if params['hinge_loss_multiplier'] is not None:
            self.hinge_loss_multiplier = params['hinge_loss_multiplier']

        if params['max_feature_deletion'] is not None:
            self.max_feature_deletion = params['max_feature_deletion']

    def get_available_params(self) -> Dict:
        params = {'hinge_loss_multiplier': self.hinge_loss_multiplier,
                  'max_feature_deletion': self.max_feature_deletion}
        return params


    def train(self):
        """
        Opitimize weight vector using FDROP algorithm
        i.e. formula (6) described in Globerson and Roweis paper
        Returns: optimized weight vector

        """
        num_instances = len(self.training_instances)

        X_neg = [ins.feature_vector for ins in self.training_instances
                 if ins.get_label() == RobustLearner.negative_classification]
        X_pos = [ins.feature_vector for ins in self.training_instances
                 if ins.get_label() == RobustLearner.positive_classification]

        y_list = y_list = [ins.label for ins in self.training_instances]



        X_list = [ins.feature_vector for ins in self.training_instances]
        y_list = [ins.label for ins in self.training_instances]

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

        self.weight_vector = sol[:self.num_features]

    def predict(self, instances: List[Instance]):
        predictions = []
        for instance in instances:
            features = instance.get_feature_vector().get_csr_matrix().toarray()[0]
            predictions.append(np.sign(self.predict_instance(features)))
        return predictions

    def predict_instance(self, features):
        return np.dot(features, self.weight_vector)

    def predict_proba(self, instances: List[Instance]):
        return [self.predict_instance(
            ins.get_feature_vector().get_csr_matrix().toarray()[0]) for ins in instances]
