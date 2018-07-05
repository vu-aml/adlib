from learners.learner import learner
from data_reader.binary_input import Instance
from data_reader.operations import sparsify
from adversaries.adversary import Adversary
from typing import List, Dict
import numpy as np
import cvxpy as cvx
from cvxpy import Variable as Variable
from cvxpy import mul_elemwise as mul
from data_reader.dataset import EmailDataset

OPT_INSTALLED = True
try:
    import cvxopt
except ImportError:
    OPT_INSTALLED = False


class SVMFreeRange(learner):
    """Solves asymmetric dual problem: :math:`argmin (1/2)*⎜⎜w⎟⎟^2 + C*∑(xi0)`

    By solving the convex optimization, optimal weight and bias matrices are
    computed to be used in the linear classification of the instances
    changed by the adversary.

    Args:
        c_f (float): aggressiveness assumption c_f ∈ [0.0,1.0]
        Note: 0.0 means no attack and 1.0 is most aggressive. Default:0.5

        xmin (float): smallest value that a feature can take. Default:0.0
        xmax (float): largest value that a feature can take. Default:1.0

    """

    def __init__(self, params=None, training_instances=None):
        learner.__init__(self)
        self.weight_vector = None
        self.bias = 0
        self.c_f = 0.5
        self.xmin = 0.0
        self.xmax = 1.0
        self.c = 0.1
        if params is not None:
            self.set_params(params)
        if training_instances is not None:
            self.set_training_instances(training_instances)

    def set_params(self, params: Dict):
        if 'c_f' in params:
            self.c_f = params['c_f']
        if 'xmin' in params:
            self.xmin = params['xmin']
        if 'xmax' in params:
            self.xmax = params['xmax']
        if 'c' in params:
            self.c = params['c']

    def get_available_params(self) -> Dict:
        params = {'c_f': self.c_f,
                  'xmin': self.xmin,
                  'xmax': self.xmax,
                  'c': self.c}
        return params

    def train(self):
        '''Optimize the asymmetric dual problem and return optimal w and b.'''
        if not self.training_instances:
            raise ValueError('Must set training instances before training')

        if isinstance(self.training_instances, List):
            y, X = sparsify(self.training_instances)
            y, X = np.array(y), X.toarray()
        else:
            X, y = self.training_instances.numpy()

        i_neg = np.array([ins[1] for ins in zip(y, X) if ins[0] ==
                          self.negative_classification])
        i_pos = np.array([ins[1] for ins in zip(y, X) if ins[0] ==
                          self.positive_classification])
        ones_col = np.ones((i_neg.shape[1], 1))
        pn = np.concatenate((i_pos, i_neg))
        pl = np.ones(i_pos.shape[0])
        nl = -np.ones(i_neg.shape[0])
        pnl = np.concatenate((pl, nl))
        xj_min = np.full_like(pn, self.xmin)
        xj_max = np.full_like(pn, self.xmax)
        ones_mat = np.ones_like(pnl)
        col_neg, row_sum = i_neg.shape[1], i_pos.shape[0] + i_neg.shape[0]

        # define cvxpy variables
        w = cvx.Variable(col_neg)
        b = cvx.Variable()
        xi0 = cvx.Variable(row_sum)
        t = cvx.Variable(row_sum)
        u = cvx.Variable(row_sum, col_neg)
        v = cvx.Variable(row_sum, col_neg)

        constraints = [xi0 >= 0,
                       xi0 >= 1 - mul(pnl, (pn * w + b)) + t,
                       t >= mul(self.c_f, (mul(xj_max - pn, v)
                                           - mul(xj_min - pn, u)) * ones_col),
                       u - v == 0.5 * (1 + pnl) * w.T,
                       u >= 0,
                       v >= 0]
        # objective
        obj = cvx.Minimize(0.5 * (cvx.norm(w)) + self.c * cvx.sum_entries(xi0))
        prob = cvx.Problem(obj, constraints)

        if OPT_INSTALLED:
            prob.solve(solver='MOSEK')
        else:
            prob.solve()

        self.weight_vector = np.asarray(w.value.T)[0]
        self.bias = b.value

    def predict(self, instances):
        """

            :param instances: could be a list of instances or a csr_matrix representation.
                   in the later case, we convert to np.array first.
            :return: a list of (1/-1)labels

            """
        predictions = []
        #list of instances
        if isinstance(instances, List):
            for instance in instances:
                features = instance.get_feature_vector().get_csr_matrix().toarray()
                predictions.append((int)(np.sign(self.predict_instance(features))))
        #single instance
        elif type(instances) == Instance:
            predictions = (int)(np.sign(self.predict_instance(
            instances.get_feature_vector().get_csr_matrix().toarray())))
        else:
        #email data set
        #return a num if there is a single instance
            for i in range(0, instances.features.shape[0]):
                instance = instances.features[i, :].toarray()
                predictions.append((int)(np.sign(self.predict_instance(instance))))
            if len(predictions) == 1:
                return predictions[0]
        return predictions

    def predict_proba(self, instances):
        return self.predict(instances)

    def predict_instance(self, instances: np.array):
        return self.weight_vector.dot(instances.T)[0] + self.bias

    #decision_function should be the distance to the hyperplane
    def decision_function(self, instances):
        predict_instances = self.weight_vector.dot(instances.T) + self.bias
        #norm = np.linalg.norm(self.weight_vector)
        return predict_instances


    def get_weight(self):
        return self.weight_vector

    def get_constant(self):
        return self.bias
