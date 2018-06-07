# data_modification.py
# An implementation of the data modification attack where the attacker modifies
# certain feature vectors
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from data_reader.real_input import RealFeatureVector
from copy import deepcopy
from typing import List, Dict
import math
import numpy as np
import pathos.multiprocessing as mp


class DataModification(Adversary):
    """
    Performs a data modification attack where the attacker can only change
    feature vectors.
    """

    def __init__(self, learner, target_theta, lda=0.001, alpha=1e-3, beta=0.05,
                 max_iter=1000, verbose=False):
        """
        :param learner: the trained learner
        :param target_theta: the theta value of which to target
        :param lda: lambda - implies importance of cost
        :param alpha: convergence condition (diff <= alpha)
        :param beta: learning rate - will be divided by size of input
        :param max_iter: maximum iterations
        :param verbose: if True, will print gradient for each iteration
        """

        Adversary.__init__(self)
        self.learner = deepcopy(learner).model.learner
        self.target_theta = target_theta
        self.lda = lda
        self.alpha = alpha
        self.beta = beta
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
        """
        Performs a data modification attack
        :param instances: the input instances
        :return: the attacked instances
        """

        if len(instances) == 0:
            raise ValueError('Need at least one instance.')

        self.instances = instances
        self.return_instances = deepcopy(self.instances)
        self._calculate_constants()

        fv_dist = 0.0
        theta_dist = np.linalg.norm(self.theta - self.target_theta)
        iteration = 0
        while (iteration == 0 or (fv_dist > self.alpha and
                                  iteration < self.max_iter)):

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

        print('Iteration: FINAL - FV distance: ', fv_dist,
              ' - theta distance: ', theta_dist, ' - alpha: ', self.alpha,
              ' - beta: ', self.beta, sep='')

        if self.verbose:
            print('\nTarget Theta:\n', self.target_theta, '\n\nTheta:\n',
                  self.theta, '\n')

        # Go from floating-point values in [0, 1] to integers in {0, 1}
        for fv in self.fvs:
            indices = []
            data = []
            for i, val in enumerate(fv):
                if val != 0:
                    indices.append(i)
                    data.append(val)

            self.return_instances[i].feature_vector = RealFeatureVector(
                self.return_instances[i].get_feature_count(), indices, data)

        return self.return_instances

    def _project_fvs(self):
        """
        Transforms all values in self.fvs to have non-negative values
        """

        for i, row in enumerate(self.fvs):
            for j, val in enumerate(row):
                if val[0] < 0:
                    self.fvs[i][j] = 0

    def _calculate_constants(self):
        """
        Calculates constants for the gradient descent loop
        """

        # Calculate feature vectors
        self.fvs = []
        for i in range(len(self.instances)):
            fv = self.instances[i].get_feature_vector().get_csr_matrix()
            fv = np.array(fv.todense().tolist()).flatten()
            self.fvs.append(fv)

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
            DataModification.logistic_function(self.labels[i] * self.g_arr[i])
            for i in range(len(self.instances))]
        self.logistic_vals = np.array(self.logistic_vals)

        # Calculate beta relative to size of input
        self.beta /= self.fvs.shape[1]

    def _calc_theta(self):
        """
        Calculates theta from learning the feature vectors
        """

        self.learner.fit(self.fvs, self.labels)  # Retrain learner
        self.theta = self.learner.decision_function(
            np.eye(self.instances[0].get_feature_count()))
        self.theta = self.theta - self.b

    def _calc_gradient(self):
        """
        Calculates the gradient of the feature vectors
        :return: the gradient
        """

        self.risk_gradient = self.theta - self.target_theta

        pool = mp.Pool(mp.cpu_count())
        matrices_1 = pool.map(self._calc_partial_f_partial_capital_d,
                              range(len(self.instances)))
        pool.close()
        pool.join()
        matrices_1 = np.array(matrices_1)

        matrix_2 = self._calc_partial_f_partial_theta()
        #  self.fuzz_matrix(matrix_2)

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
        """
        Calculates ∂f/∂Θ
        :return: the partial derivative
        """

        pool = mp.Pool(mp.cpu_count())
        matrix = pool.map(lambda j: list(map(
            lambda k: self._calc_partial_f_j_partial_theta_k(j, k),
            range(len(self.theta)))), range(len(self.theta)))
        pool.close()
        pool.join()

        return np.array(matrix)

    def _calc_partial_f_j_partial_theta_k(self, j, k):
        """
        Calculates ∂f_j / ∂Θ_k
        :param j: see above
        :param k: see above
        :return: the partial derivative
        """

        running_sum = 0.0
        for i in range(len(self.instances)):
            val = self.logistic_vals[i]
            running_sum += self.fvs[i][k] * self.fvs[i][j] * val * (1 - val)
        running_sum += self.lda if j == k else 0
        return running_sum

    def _calc_partial_f_partial_capital_d(self, i):
        """
        Calculates ∂f/∂D
        :param i: indicates which feature vector to use
        :return: the partial derivative
        """

        matrix = list(map(lambda j: list(map(
            lambda k: self._calc_partial_f_j_partial_x_k(i, j, k),
            range(len(self.theta)))), range(len(self.theta))))

        return np.array(matrix)

    def _calc_partial_f_j_partial_x_k(self, i, j, k):
        """
        Calculates ∂f_j / ∂x_k
        :param i: indicates which feature vector to use
        :param j: see above
        :param k: see above
        :return: the partial derivative
        """

        val = self.logistic_vals[i]
        return ((1 - val) * (val * self.theta[k] * self.fvs[i][j] -
                             self.labels[i] if j == k else 0))

    @staticmethod
    def fuzz_matrix(matrix: np.ndarray):
        """
        Add to every entry of matrix some noise to make it non-singular.
        :param matrix: the matrix - 2 dimensional
        """

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                matrix[i][j] += abs(np.random.normal(0, 0.00001))

    @staticmethod
    def logistic_function(x):
        """
        :param x: x
        :return: the logistic function of x
        """

        return 1 / (1 + math.exp(-1 * x))

    def set_params(self, params: Dict):
        if params['learner'] is not None:
            self.learner = params['learner']
        if params['target_theta'] is not None:
            self.target_theta = params['target_theta']
        if params['lda'] is not None:
            self.lda = params['lda']
        if params['alpha'] is not None:
            self.alpha = params['alpha']
        if params['beta'] is not None:
            self.beta = params['beta']
        if params['max_iter'] is not None:
            self.max_iter = params['max_iter']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.instances = None
        self.return_instances = None
        self.orig_fvs = None
        self.old_fvs = None
        self.fvs = None
        self.theta = None
        self.b = None
        self.g_arr = None
        self.labels = None
        self.logistic_vals = None
        self.risk_gradient = None

    def get_available_params(self):
        params = {'learner': self.learner,
                  'target_theta': self.target_theta,
                  'lda': self.lda,
                  'alpha': self.alpha,
                  'beta': self.beta,
                  'max_iter': self.max_iter,
                  'verbose': self.verbose}
        return params

    def set_adversarial_params(self, learner, train_instances):
        self.learner = learner
        self.instances = train_instances
