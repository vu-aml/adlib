from learners.learner import learner
from data_reader.binary_input import Instance
from data_reader.operations import sparsify
from typing import List, Dict
import numpy as np
from cvxpy import *


class FeatureDeletion(learner):
    def __init__(self, training_instances=None, params=None):

        learner.__init__(self)
        self.weight_vector = None  # type: np.array(shape=(1))
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

        C = self.hinge_loss_multiplier
        print("current C value: {}".format(C))
        print("current K: {}".format(self.max_feature_deletion))
        print(X.shape)
        K = self.max_feature_deletion

        w = Variable(self.num_features)  # weight vector
        b = Variable()  # bias term
        t = Variable(num_instances)
        z = Variable(num_instances)
        v = Variable(num_instances, self.num_features)

        loss = sum_entries(pos(1 - mul_elemwise(y, X * w + b) + t))

        constraints = [t >= K * z + sum_entries(v, axis=1),
                       v >= 0]
        constraints.extend([z[i] + v[i, :] >=
                            y[i] * mul_elemwise(X[i], w).T for i in range(num_instances)])
        obj = Minimize(0.5 * (sum_squares(w)) + C * loss)

        prob = Problem(obj, constraints)
        prob.solve(solver=SCS)
        # print("training completed, here is the learned weight vector:")

        # weight_vector is of shape (1, self.num_features)
        self.weight_vector = [np.array(w.value).T][0]
        self.bias = b.value

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
        return self.weight_vector.dot(features.T)[0][0] + self.bias

    def decision_function(self):
        return self.weight_vector, self.bias

    def get_weight(self):
        return self.weight_vector

    def get_constant(self):
        return self.bias
