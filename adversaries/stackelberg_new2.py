from data_reader.input import Instance
from data_reader.operations import sparsify
from typing import List
import numpy as np
from scipy import optimize
from scipy import special

'''Stackelberg game of complete information between adversary and improved learner.

Concept:
	Given loss functions for the learner and adversary and the costs for each
	instance in the training set, finds the transformation of the instances and
	the predictive model for the learner. Solution is optimal in a multi-stage game,
	in which the leader (learner) trains, and the follower (adversary) disrupts the data.
	Adversary has full information about the learner, and is assumed to act in a rational
	(adversarially-optimal) way. Since the learner knows the information given to the
	adversary, as well as the fact that it will try to optimize its payoff, the learner
	is able to pre-empt the adversary, creating an equilibrium solution.

'''



class SPG(object):
    LINEAR_LOSS = 'linear_loss'
    WORST_CASE_LOSS = 'worst_case_loss'
    LOGISTIC_LOSS = 'logistic_loss'

    def __init__(self, adversary_costs, learner_costs,
                 adversary_regularization_factor, learner_regularization_factor, train_instances, max_iter=1000):
        """

        :param adversary_costs:  c+1 of type nparray
        :param learner_costs:   c-1 of type nparray
        :param adversary_regularization_factor:  rho +1
        :param learner_regularization_factor: rho-1
        :param max_iter: maximum iteration for solver
        :param train_instances: Training instances of type List[Instance]
        """
        self.weight_vector = None # type: np.array() , shape= (1,num_features)
        self.tau_factors = None

        self.adversary_costs = adversary_costs  # type: np.array
        self.learner_costs = learner_costs  # type: np.array
        self.adversary_regularization_factor = adversary_regularization_factor  # type: float
        self.learner_regularization_factor = learner_regularization_factor  # type: float
        self.max_iteration = max_iter   # type: int
        self.training_instances = train_instances  # type: List[Instance]
        self.num_features = self.training_instances[0].get_feature_vector().get_feature_count()

    def optimize(self):
        raise NotImplementedError

    def loss_function(self, z: float, y: int):
        raise NotImplementedError


class LogisticLoss(SPG):

    def optimize(self):
        num_instances = len(self.training_instances)
        x0 = np.array([0.0] * (num_instances + self.num_features))
        y_list, X_list = sparsify(self.training_instances)
        y, X = np.array(y_list), X_list.toarray()
        c_l = self.learner_costs
        c_a = self.adversary_costs
        rho_l = self.learner_regularization_factor
        rho_a = self.adversary_regularization_factor

        cons = [{'type': 'eq',
                 'fun': logistic_constraints,
                 'args': (i, num_instances, self.num_features, X, c_a,rho_a)}
                for i in range(num_instances)]
        print('begin scipy minimize')
        result = optimize.minimize(logistic_objective_func, x0,
                                    args=(self.num_features, X, y, c_l, rho_l),method='SLSQP', constraints=cons,
                                    options={'maxiter':self.max_iteration})
        print('minimize finished')
        if not result.success:
            raise ValueError("optimize failed")
        self.weight_vector, self.tau_factors = np.array(result.x[:self.num_features]).reshape((1,self.num_features)),\
                                               np.array(result.x[self.num_features:])


def logistic_objective_func(x0, num_features=0, X=None, y=None, c_l=None,rho_l=0):
    """

    :param x0:
    :param num_features:
    :param X: numpy array of shape num_instances * num_features
    :param y: numpy array of shape (num_instances,)
    :param c_l:
    :param rho_l:
    :return:
    """
    w, tau = np.asarray(x0[:num_features]), np.asarray(x0[num_features:])
    norm_squared = np.sum(w**2)
    sum = np.sum(c_l*special.expit((w.T.dot(X.T)+norm_squared*tau)*y))
    reg = rho_l/2 * norm_squared
    return sum + reg


def logistic_constraints(x0, i=0, num_instances=0, num_features=0, X=None, c_a=None, rho_a=0):
    """

    :param x0:
    :param num_instances:
    :param num_features:
    :param X:
    :param c_a:
    :param rho_a:
    :return:
    """
    w, tau = np.asarray(x0[:num_features]), np.asarray(x0[num_features:])
    norm_squared = np.sum(w**2)

    return tau[i]*(1+np.exp((-1)*w.T.dot(X[i])-tau[i]*norm_squared)) + (num_instances/rho_a)*c_a[i]


def loss_function(self, z:np.ndarray, y:np.ndarray):
    """
    scipy.special.expit
    for logistic lost
    """
    return special.expit(z*y)
