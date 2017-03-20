from learners.learner import RobustLearner
from data_reader.input import Instance, FeatureVector
from data_reader.operations import sparsify
from adversaries.adversary import Adversary
from typing import List, Dict
import numpy as np
import cvxpy as cvx
from cvxpy import Variable as Variable
from cvxpy import mul_elemwise as mul
OPT_INSTALLED = True
try:
    import cvxopt
except ImportError:
    OPT_INSTALLED = False



class SVMRestrained(RobustLearner):
    """Solves asymmetric dual problem: :math:`argmin (1/2)*⎜⎜w⎟⎟^2 + C*∑(xi0)`

    By solving the convex optimization, optimal weight and bias matrices are
    computed to be used in the linear classification of the instances
    changed by the adversary.

    Args:
        c_delta: aggressiveness assumption c_delta ∈ [0.0,1.0]. Default:0.5
    """

    def __init__(self, params = None, training_instances = None):
        RobustLearner.__init__(self)
        self.weight_vector = None
        self.bias = 0
        self.c_delta = 0.5
        if params is not None:
            self.set_params(params)
        if training_instances is not None:
            self.set_training_instances(training_instances)


    def set_params(self, params: Dict):
        if 'c_delta' in params:
            self.c_delta = params['c_delta']

    def get_available_params(self) -> Dict:
        params = {'c_delta': self.c_delta}
        return params

    def train(self):
        c = 10
        num_instances = len(self.training_instances)
        y,X = sparsify(self.training_instances)
        y,X = np.array(y), X.toarray()
        i_neg = np.array([ins[1] for ins in zip(y,X) if ins[0]==self.negative_classification])
        i_pos = np.array([ins[1] for ins in zip(y,X) if ins[0]==self.positive_classification])

        # centroid can be computed in multiple ways
        n_centroid = np.mean(i_neg)
        Mk = ((1-self.c_delta * np.fabs(n_centroid - i_pos)/
              (np.fabs(n_centroid)+np.fabs(i_pos)))*
              ((n_centroid-i_pos)**2))
        Zks = np.zeros_like(i_neg)
        Mk = np.concatenate((Mk,Zks))
        TMk = np.concatenate((n_centroid - i_pos,Zks))
        ones_col = np.ones((i_neg.shape[1],1))
        pn = np.concatenate((i_pos,i_neg))
        pl = np.ones(i_pos.shape[0])
        nl = -np.ones(i_neg.shape[0])
        pnl = np.concatenate((pl,nl))
        col_neg, row_sum = i_neg.shape[1], i_pos.shape[0] + i_neg.shape[0]

        # constraint cvxpy variables
        w = Variable(col_neg)
        b = Variable()
        xi0 = Variable(row_sum)
        t = Variable(row_sum)
        u = Variable(row_sum,col_neg)
        v = Variable(row_sum,col_neg)

        constraints = [xi0>=0,
                       xi0 >= 1-mul(pnl, (pn*w+b))+t,
                       t >= mul(Mk,u)*ones_col,
                       mul(TMk,(-u+v))==0.5*(1+pnl)*w.T,
                       u>=0,
                       v>=0]

        # Objective
        obj = cvx.Minimize(0.5*(cvx.norm(w)) + c*cvx.sum_entries(xi0))
        prob = cvx.Problem(obj,constraints)

        if OPT_INSTALLED:
            prob.solve(solver='CVXOPT')
        else:
            prob.solve()

        self.weight_vector = [np.array(w.value).T][0]
        self.bias = b.value

    def predict(self, instances: List[Instance]):
        predictions = []
        for instance in instances:
            features = instance.get_feature_vector().get_csr_matrix().toarray()
            predictions.append(np.sign(self.predict_instance(features)))
        return predictions

    def predict_instance(self, features):
        return self.weight_vector.dot(features.T)[0][0] + self.bias

    def predict_proba(self, instances: List[Instance]):
        return [self.predict_instance(
            ins.get_feature_vector().get_csr_matrix().toarray()) for ins in instances]

    def decision_function(self):
        return self.weight_vector, self.bias
