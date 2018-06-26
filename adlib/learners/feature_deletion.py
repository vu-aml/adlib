from adlib.learners.learner import Learner
from data_reader.binary_input import Instance
from data_reader.operations import sparsify
from typing import List, Dict
import numpy as np
from cvxpy import *


class FeatureDeletion(Learner):
    def __init__(self, training_instances=None, params=None):

        Learner.__init__(self)
        self.weight_vector = None  # type: np.array(shape=1)
        self.num_features = 0  # type: int
        self.hinge_loss_multiplier = 0.5  # type: float
        self.max_feature_deletion = 30  # type: int
        self.bias = 0  # type: int
        if params is not None:
            self.set_params(params)
        if training_instances is not None:
            self.set_training_instances(training_instances)

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
        if isinstance(self.training_instances, List):
            y_list, X_list = sparsify(self.training_instances)
            num_instances = len(y_list)
            y, X = np.array(y_list).reshape((num_instances, 1)), X_list.toarray().reshape(
                (num_instances, self.num_features))

        else:
            X, y = self.training_instances.numpy()
            num_instances = len(y)
            y, X = y.reshape((num_instances, 1)), X

        # append another column at X for bias
        bias_col = np.ones_like(y.T)
        X_prime = np.insert(X, X.shape[1], bias_col, axis=1)

        C = self.hinge_loss_multiplier
        print("current C value(hinge loss multipler): {}".format(C))
        print("current K(maximum feature deletion): {}".format(self.max_feature_deletion))
        print(X.shape)
        print(y.shape)
        K = self.max_feature_deletion
        w = Variable(self.num_features + 1)  # weight vector
        # b = Variable()  # bias term
        t = Variable(num_instances)
        z = Variable(num_instances)
        v = Variable(num_instances, (self.num_features + 1))
        loss_f = Variable(num_instances)

        # bias is implemented as the last column of X(a extra feature vector of only 1)
        # bias in the weight vector is not calculated in the regularization

        constraints = [t >= K * z + sum_entries(v, axis=1), v >= 0]
        constraints.extend([z[i] + v[i, :] >=
                            y[i] * mul_elemwise(X_prime[i], w).T for i in range(num_instances)])
        constraints.extend([loss_f[i] >= 0 for i in range(num_instances)])
        constraints.extend([loss_f[i] >= (1 - y[i] * (X_prime[i] * w) + t[i]) for i in range(num_instances)])
        obj = Minimize(0.5 * (sum_squares(w[:-1])) + C * sum_entries(loss_f))

        #  constraints = [t >= K * z + sum_entries(v, axis=1),v >= 0]
        #  constraints.extend([z[i] + v[i, :] >=
        #                      y[i] * mul_elemwise(X[i], w) for i in range(num_instances)])
        #  constraints.extend([])
        #  constraints.extend([loss_f[i] >= 0 for i in range(num_instances)])
        #  constraints.extend([loss_f[i] >= (1 - y[i] * (X[i] * w + b) + t[i])
        #  for i in range(num_instances)])
        #  obj = Minimize(0.5 * (sum_squares(w)) + C * sum_entries(loss_f))

        prob = Problem(obj, constraints)
        # switch another server to solve the scalability issue
        prob.solve(solver=SCS)
        # print("training completed, here is the learned weight vector:")
        # weight_vector is of shape (1, self.num_features)
        self.weight_vector = [np.array(w.value).T][0][0][:-1]
        self.bias = [np.array(w.value).T][0][0][-1]
        self.t = t
        print(self.weight_vector)
        print(self.bias)

        print("final weight vector shape: {}".format(self.weight_vector.shape))
        # print("bias term:{}".format(self.bias))
        top_idx = [i for i in np.argsort(np.absolute(self.weight_vector))[-10:]]
        print("indices with top 10 absolute value:")
        for i in top_idx:
            print("index No.{} with value {}".format(i, self.weight_vector[i]))

    def predict(self, instances):
        """

         :param instances: matrix of instances shape (num_instances,
                           num_feautres_per_instance)
         :return: list of int labels
         """
        predictions = []
        # list of instances
        if isinstance(instances, List):
            for instance in instances:
                features = instance.get_feature_vector().get_csr_matrix().toarray()
                predictions.append(np.sign(self.predict_instance(features)))
        # single instance
        elif type(instances) == Instance:
            predictions = np.sign(self.predict_instance(
                            instances.get_feature_vector().get_csr_matrix().toarray()))
        else:
            predictions = []
            for i in range(0, instances.features.shape[0]):
                instance = instances.features[i, :].toarray()
                predictions.append(np.sign(self.predict_instance(instance)))
            if len(predictions) == 1:
                return predictions[0]
        return predictions

    def predict_instance(self, features):
        '''
        predict class for a single instance and return a real value
        :param features: np.array of shape (1, self.num_features),
                         i.e. [[1, 2, ...]]
        :return: float
        '''
        return self.weight_vector.dot(features.T)[0] + self.bias

    # decision_function should be the distance to the hyperplane
    def decision_function(self, instances):
        predict_instances = self.weight_vector.dot(instances.T) + self.bias
        norm = np.linalg.norm(self.weight_vector)
        return predict_instances / norm

    def get_weight(self):
        return self.weight_vector

    def get_constant(self):
        return self.bias
