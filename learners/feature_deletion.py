from learners.learner import RobustLearner
from data_reader.input import Instance
from data_reader.operations import sparsify
from typing import List, Dict
import numpy as np
from cvxpy import *


class FeatureDeletion(RobustLearner):

    def __init__(self, params=None, training_instances=None):
        RobustLearner.__init__(self)
        self.weight_vector = None        # type: np.array(shape=(1))
        self.num_features = 0  # type: int
        self.hinge_loss_multiplier = 0.5  # type: float
        self.max_feature_deletion = 30  # type: int

        self.weight_vector = None   # type: np.array
                                    # of shape (1, self.num_features)
        self.bias = 0  # type: int

        self.set_params(params)
        self.set_training_instances(training_instances)
        print(self.num_features)

    def set_params(self, params: Dict):
        if 'hinge_loss_multiplier' in params:
            self.hinge_loss_multiplier = params['hinge_loss_multiplier']

        if 'max_feature_deletion' in params:
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

        y_list, X_list = sparsify(self.training_instances)
        y, X = np.array(y_list), X_list.toarray()

        C = self.hinge_loss_multiplier
        K = self.max_feature_deletion


        i_ones, i_zeroes, j_ones = np.ones(num_instances), np.zeros(num_instances), np.ones(self.num_features)
        w = Variable(self.num_features)  # weight vector
        b = Variable()  # bias term
        t = Variable(num_instances)
        z = Variable(num_instances)
        v = {}
        loss = sum_entries(pos(1 - mul_elemwise(y, X * w + b) + t))  # loss func
        for i in range(num_instances):
            v[i] = Variable(self.num_features)
        yX = y.dot(X)

        constraints = [t >= K * z + np.dot(X, j_ones)]

        # add constraints vi >= 0
        constraints.extend([v[i] >= 0 for i in range(num_instances)])

        # add constraints zi + vi >= y.dot(X) * w
        constraints.extend([v[i][j] + z[i] >= yX[j]*w[j] for j in range(self.num_features)
                                     for i in range(num_instances)])

        obj = Minimize(0.5*(norm(w)) + C * loss)

        prob = Problem(obj, constraints)
        prob.solve()

        self.weight_vector = [np.array(w.value).T][0]   # weight_vector is of shape (1, self.num_features)
        self.bias = b.value


    def predict(self, instances: List[Instance]):
        predictions = []
        for instance in instances:
            features = instance.get_feature_vector().get_csr_matrix().toarray()
            predictions.append(np.sign(self.predict_instance(features)))
        return predictions

    def predict_instance(self, features):
        '''
        predict class for a single instance and return a real value
        :param features: np.array of shape (1, self.num_features), i.e. [[1, 2, ...]]
        :return: float
        '''
        print('w shape: '+ str(self.weight_vector.shape))
        return self.weight_vector.dot(features.T)[0][0] + self.bias

    def predict_proba(self, instances: List[Instance]):
        return [self.predict_instance(
            ins.get_feature_vector().get_csr_matrix().toarray()) for ins in instances]

    def decision_function(self):
        return self.weight_vector, self.bias