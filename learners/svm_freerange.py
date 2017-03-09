from learners.learner import RobustLearner
from data_reader.input import Instance, FeatureVector
from typing import List, Dict
import numpy as np
from numpy.matlib import repmat
import cvxpy as cvx

OPT_INSTALLED = True
try:
    import cvxopt
except ImportError:
    OPT_INSTALLED = False

class SVMFreeRange(RobustLearner):
    """Solves asymmetric dual problem: :math:`argmin (1/2)*⎜⎜w⎟⎟^2 + C*∑(xi0)`

    By solving the convex optimization, optimal weight and bias matrices are
    computed to be used in the linear classification of the instances
    changed by the adversary.

    Args:
        c_f (float): aggressiveness assumption c_f ∈ [0.0,1.0]
        Note: 0.0 means no attack and 1.0 is most aggressive. Default:0.5

        x_min (float): smallest value that a feature can take. Default:0.0
        x_max (float): largest value that a feature can take. Default:1.0

    """

    def __init__(self):
        RobustLearner.__init__(self)
        self.weight_vector = None

    def set_params(self, params: Dict):
        self.c = 10
        # self.atk_f = 0.5 # adversary parameter
        #TODO: add these to __init__:
        self.c_f = 0.5 # this parameter can be tweaked
        self.x_min = 0.0
        self.x_max = 1.0

        if params['instances'] is not None:
            self.neg_i = np.fromiter(i for i in instances if \
                         i.get_label() == InitialPredictor.negative_classification)
            self.pos_i = np.fromiter(i for i in instances if \
                         i.get_label() == InitialPredictor.positive_classification)
            self.ones_col = np.ones((self.neg_i.shape[1],1))
            self.pn = np.stack((self.pos_i,self.neg_i))
            self.pl = np.ones((self.pos_i[0],1))
            self.nl = -np.ones((self.neg_i[0],1))
            self.pnl = np.stack((self.pl,self.nl))
            self.xj_min = np.empty_like(self.pn).fill(self.x_min)
            self.xj_max = np.empty_like(self.pn).fill(self.x_max)
            self.ones_mat = np.ones_like(self.pnl)
            #self.set_adversarial_params()

            params = self.cvx_optimize(self.neg_i.size[1],self.pos_i.size[0]+self.neg_i.size[0])
            self.weight_vector = params.weight
            self.bias = params.bias

    def cvx_optimize(self, col_neg: int, row_sum: int):
        """Optimize the asymmetric dual problem and return optimal w and b.

        Args:
            col_neg: int number of columns of negative instances
            row_sum: int sum of rows of negative and positive instances
        """

        w = cvx.Variable(col_neg)
        b = cvx.Variable()
        xi0 = cvx.Variable(row_sum)
        t = cvx.Variable(row_sum)
        u = cvx.Variable(row_sum,col_neg)
        v = cvx.Variable(row_sum,col_neg)

        constraints = [xi0>=0,
                       xi0 >=1-self.pnl*(np.dot(self.pn,w)+b)+t,
                       t>=self.c_f*np.dot((v*(self.xj_max-self.pn) - u*(self.xj_min - self.pn)),self.ones_col),
                       u-v==0.5*repmat((1+self.pnl),1,col_neg)*repmat(w.T,row_sum,1),
                       u>=0,
                       v>=0]
        # objective
        # TODO: Test to see if this should be cvx.sum()
        obj = cvx.Minimize(0.5*(cvx.norm(w)) + self.c*cvx.sum_entries(xi0))
        prob = cvx.Problem(obj,constraints)
        if OPT_INSTALLED:
            prob.solve(solver=CVXOPT)
        else:
            prob.solve()
        return {'weight':w.value,'bias':b.value}

    def train(self):
        # this needs to be implemented
        raise NotImplementedError

    def predict(self, instances: List[Instance]):
        predictions = []
        for instance in instances:
            features = instance.get_feature_vector().get_csr_matrix().toarray()[0]
            predictions.append(np.sign(self.predict_instance(features)))
        return predictions

    def predict_instance(self, features: Instance):
        return np.dot(features, self.weight_vector) + self.bias
