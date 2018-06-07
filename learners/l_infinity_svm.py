from cvxpy import *
import numpy as np
from learners.learner import learner
from data_reader.binary_input import Instance
from data_reader.operations import sparsify
from typing import List, Dict

class L_infSVM(learner):
    """
    L-infinity Regularized Support Vector Machine with soft margin and linear kernel

    Optimization process is done using cvxpy interface, which internally calls ECOS
    solver. ECOS is a numerical software for solving convex second-order cone programs (SOCPs)
    of type:
        min  c'*x
        s.t. A*x = b
        G*x <=_K h

    """


    def __init__(self, training_instances=None, coef=0.25):

        learner.__init__(self)
        self.weight_vector = None  # type: np.array(shape=(1))
        self.num_features = 0  # type: int
        self.reg_coef = coef  # type: float
        self.bias = 0  # type: int

        if training_instances is not None:
            self.set_training_instances(training_instances)

    def train(self):
        """
        train classifier based on training data and corresponding label
        :param X: training feature vectors in matrix form
        :param y: corresponding labels in array
        :return: None
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

        n, m = len(X[0]), len(X)

        weights = Variable(n)
        bias = Variable()
        loss_func = sum_entries(pos(1 - mul_elemwise(y, X*weights - bias)))
        reg_term = norm(weights, 'inf')
        reg_factor = self.reg_coef

        # define optimization problem
        prob = Problem(Minimize(loss_func / m + reg_factor * reg_term))
        prob.solve()
        self.weight_vector = weights.value
        self.bias = bias.value

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
                predictions.append(self.predict_instance(features))
        # single instance
        elif type(instances) == Instance:
            predictions = self.predict_instance(
                            instances.get_feature_vector().get_csr_matrix().toarray())
        else:
            for i in range(0, instances.features.shape[0]):
                instance = instances.features[i, :].toarray()
                predictions.append(self.predict_instance(instance))
            if len(predictions) == 1:
                return predictions[0]
        return predictions

    def predict_instance(self, features):
        '''
        predict class for a single instance and return a real value
        :param features: np.array of shape (1, self.num_features),
                         i.e. [[1, 2, ...]]
        :return: int
        '''
        return int(np.sign(np.asscalar(features.dot(self.weight_vector)) + self.bias))

    #decision_function should be the distance to the hyperplane
    def decision_function(self, instances):
        predict_instances = self.weight_vector.dot(instances.T) + self.bias
        norm = np.linalg.norm(self.weight_vector)
        return predict_instances / norm

    def get_weight(self):
        return np.asarray(self.weight_vector.T)[0]
