# data_modification.py
# An implementation of the data modification attack where the attacker modifies
# certain feature vectors
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import BinaryFeatureVector, Instance
from copy import deepcopy
from typing import List, Dict
import math
import numpy as np
import pathos.multiprocessing as mp


class DataModification(Adversary):
    def __init__(self, learner, target_theta, lda=0.5, alpha=1e-5, beta=1,
                 max_iter=10000, verbose=False):

        Adversary.__init__(self)
        self.learner = deepcopy(learner).model.learner
        self.target_theta = target_theta
        self.lda = lda  # lambda
        self.alpha = alpha
        self.beta = beta  # learning rate - will be divided by size of input
        self.max_iter = max_iter
        self.verbose = verbose
        self.instances = None
        self.return_instances = None
        self.orig_fvs = None  # same as below, just original values
        self.old_fvs = None  # last iteration's fvs values
        self.fvs = None  # feature vector matrix, shape: (# inst., # features)
        self.theta = None
        self.b = None
        self.g_arr = None  # array of g_i values, shape: (# inst.)
        self.labels = None  # array of labels of the instances
        self.logistic_vals = None
        self.risk_gradient = None

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0:
            raise ValueError('Need at least one instance.')

        self.instances = instances
        self.return_instances = deepcopy(self.instances)
        self._calculate_constants()

        fv_dist = 0.0
        theta_dist = np.linalg.norm(self.theta - self.target_theta)
        iteration = 0
        while (iteration == 0 or (theta_dist > self.alpha and
                                  iteration < self.max_iter)):

            if self.verbose:
                print('Iteration: ', iteration, ' - FV distance: ', fv_dist,
                      ' - theta distance: ', theta_dist, sep='')

            # Gradient descent
            gradient = self._calc_gradient()

            if self.verbose:
                print('\nGRADIENT\n', gradient, '\n', sep='')

            self.fvs -= (gradient * self.beta)
            self._project_fvs()

            # Update variables
            self._calc_theta()
            fv_dist = np.linalg.norm(self.fvs - self.old_fvs)
            theta_dist = np.linalg.norm(self.theta - self.target_theta)
            self.old_fvs = deepcopy(self.fvs)

            iteration += 1

        if self.verbose:
            print('Iteration: FINAL - FV distance: ', fv_dist,
                  ' - theta distance: ', theta_dist, ' - alpha: ', self.alpha,
                  ' - beta: ', self.beta, sep='')
            print('\nTarget Theta:\n', self.target_theta, '\n\nTheta:\n',
                  self.theta, '\n')

        for i in range(len(self.fvs)):
            indices = []
            for j in range(len(self.fvs[i])):
                if self.fvs[i][j] >= 0.5:
                    indices.append(j)
            self.return_instances[i].feature_vector = BinaryFeatureVector(
                self.return_instances[i].get_feature_count(), indices)

        return self.return_instances

    def _fuzz_matrix(self, matrix: np.ndarray):
        """
        Add to every entry of matrix some noise to make it non-singular.
        :param matrix: the matrix - 2 dimensional
        """

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] += abs(np.random.normal(0, 0.00001))

    def _project_fvs(self):
        """
        Transform all feature vectors in self.fvs from having elements in the
        interval [MIN, MAX] to the interval [0, 1]
        """

        pool = mp.Pool(mp.cpu_count())
        self.fvs = pool.map(self._project_feature_vector, self.fvs.tolist())
        pool.close()
        pool.join()

        self.fvs = np.array(self.fvs)

    def _project_feature_vector(self, fv):
        fv = np.array(fv)
        min_val = np.min(fv)
        max_val = np.max(fv)
        distance = max_val - min_val

        if distance > 0 and (min_val < 0 or max_val > 1):
            transformation = lambda x: (x - min_val) / distance
            for i in range(len(fv)):
                fv[i] = transformation(fv[i])

        return fv

    def _calculate_constants(self):
        # Calculate feature vectors as np.ndarrays
        self.fvs = []
        for i in range(len(self.instances)):
            feature_vector = self.instances[i].get_feature_vector()
            tmp = []
            for j in range(self.instances[0].get_feature_count()):
                if feature_vector.get_feature(j) == 1:
                    tmp.append(1)
                else:
                    tmp.append(0)
            tmp = np.array(tmp)
            self.fvs.append(tmp)
        self.fvs = np.array(self.fvs, dtype='float64')
        self.orig_fvs = deepcopy(self.fvs)
        self.old_fvs = deepcopy(self.fvs)

        # Calculate labels
        self.labels = []
        for inst in self.instances:
            self.labels.append(inst.get_label())
        self.labels = np.array(self.labels)

        # Calculate theta, b, and g_arr
        self.g_arr = self.learner.decision_function(self.fvs)
        self.b = self.learner.intercept_[0]
        self._calc_theta()

        # Calculate logistic function values - sigma(y_i g_i)
        self.logistic_vals = [
            DataModification._logistic_function(self.labels[i] * self.g_arr[i])
            for i in range(len(self.instances))]
        self.logistic_vals = np.array(self.logistic_vals)

        # Calculate beta relative to size of input
        self.beta /= self.fvs.shape[1]

    def _calc_theta(self):
        self.learner.fit(self.fvs, self.labels)  # Retrain learner
        self.theta = self.learner.decision_function(
            np.eye(self.instances[0].get_feature_count()))
        self.theta = self.theta - self.b

    def _calc_gradient(self):
        self.risk_gradient = self.theta - self.target_theta

        pool = mp.Pool(mp.cpu_count())
        matrices_1 = pool.map(self._calc_partial_f_partial_capital_d,
                              range(len(self.instances)))
        pool.close()
        pool.join()
        matrices_1 = np.array(matrices_1)

        matrix_2 = self._calc_partial_f_partial_theta()
        #  self._fuzz_matrix(matrix_2)

        try:
            matrix_2 = np.linalg.inv(matrix_2)
        except np.linalg.linalg.LinAlgError:
            # Singular matrix -> do not move values with this part of gradient
            print('SINGULAR MATRIX ERROR')
            matrix_2 = np.full(matrix_2.shape, 0)

        gradient = []
        for i in range(len(matrices_1)):
            partial_theta_partial_capital_d = -1 * matrix_2.dot(matrices_1[i])
            value = self.risk_gradient.dot(partial_theta_partial_capital_d)
            gradient.append(value)
        gradient = np.array(gradient)

        # Calculate cost part
        cost_gradient = self.lda * (self.fvs - self.orig_fvs)
        gradient += cost_gradient

        return gradient

    def _calc_partial_f_partial_theta(self):
        pool = mp.Pool(mp.cpu_count())
        matrix = pool.map(lambda j: list(map(
            lambda k: self._calc_partial_f_j_partial_theta_k(j, k),
            range(len(self.theta)))), range(len(self.theta)))
        pool.close()
        pool.join()

        return np.array(matrix)

    def _calc_partial_f_j_partial_theta_k(self, j, k):
        running_sum = 0.0
        for i in range(len(self.instances)):
            val = self.logistic_vals[i]
            running_sum += self.fvs[i][k] * self.fvs[i][j] * val * (1 - val)
        running_sum += self.lda if j == k else 0
        return running_sum

    def _calc_partial_f_partial_capital_d(self, i):
        matrix = list(map(lambda j: list(map(
            lambda k: self._calc_partial_f_j_partial_x_k(i, j, k),
            range(len(self.theta)))), range(len(self.theta))))

        return np.array(matrix)

    def _calc_partial_f_j_partial_x_k(self, i, j, k):
        val = self.logistic_vals[i]
        return ((1 - val) * (val * self.theta[k] * self.fvs[i][j] -
                             self.labels[i] if j == k else 0))

    @staticmethod
    def _logistic_function(x):
        return 1 / (1 + math.exp(-1 * x))

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
