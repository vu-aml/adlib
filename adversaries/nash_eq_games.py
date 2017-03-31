from typing import List
import numpy as np
from scipy import optimize
from scipy.misc import derivative
from copy import deepcopy
from util import transform_instance, partial_derivative, predict_instance

'''Nash equilibrium games.

Concept:
    Given loss functions for the learner and adversary and the costs for each
    instance in the training set, find the pair of vectors (w+, w-) such that the
    adversary transforms the data over w+ and the learner finds a decision value
    for each instance using w-. Generates the pair (w+, w-) such that learner and
    adversary reach their nash equilibrium strategy sets; neither can benefit by
    straying from the pairing.

    Solution is generated in one round (see stackelberg for multi-round).

'''

class Game(object):

    def __init__(self, loss_function, learner_loss_function, train_instances,
                 num_features, epsilon, lambda_val, learner_lambda_val):
        self.loss_function = loss_function
        self.learner_loss_function = learner_loss_function
        self.train_instances = train_instances
        self.num_features = num_features
        self.epsilon = epsilon
        self.lambda_val = lambda_val
        self.learner_lambda_val = learner_lambda_val

    def regularizer(self, a_adversary: np.array):
        return self.lambda_val/2 * np.linalg.norm(a_adversary)

    def learner_regularizer(self, a_learner: np.array):
        return self.learner_lambda_val/2 * np.linalg.norm(a_learner)

    def find_params(self):
        raise NotImplementedError

class ConvexLossEquilibirum(Game):

    def find_params(self):
        a_adversary = np.zeros(self.num_features)
        a_learner = np.zeros(self.num_features)
        k = 0
        b = self.find_min_b().fun
        r = (((self.lambda_val + self.learner_lambda_val)*self.lambda_val - 2*b -
            2*np.sqrt(b**2 - self.lambda_val*self.learner_lambda_val*b)) /
            ((self.lambda_val + self.learner_lambda_val)**2 - 4*b)) / 2

        while True:
            b_result = self.arg_max_costs(r, a_adversary, a_learner)
            (b_max_cost, b_adversary, b_learner) = b_result

            d_adversary = np.subtract(b_adversary, a_adversary)
            d_learner = np.subtract(b_learner, a_learner)

            t = self.max_step(r, a_adversary, a_learner, d_adversary, d_learner, b_max_cost)
            a_new_adversary = np.add(a_adversary, np.multiply(t, d_adversary))
            a_new_learner = np.add(a_learner, np.multiply(t, d_learner))

            difference = np.linalg.norm(np.concatenate(([np.subtract(a_new_learner, a_learner)],
                                                        [np.subtract(a_new_adversary, a_adversary)])))
            a_adversary = deepcopy(a_new_adversary)
            a_learner = deepcopy(a_new_learner)

            if difference <= self.epsilon:
                break
        return {'adversary': a_adversary,
                'learner': a_learner}

    def arg_max_costs(self, r, a_adversary, a_learner):
        x0 = [0.1] * self.num_features*2
        cost = optimize.fmin_slsqp(self.psi, x0, args=(r, a_adversary, a_learner), iter=5, iprint=2, full_output=True)
        b_adversary = cost[0][0:self.num_features]
        b_learner = cost[0][self.num_features:self.num_features*2]
        return (-cost[1], b_adversary, b_learner)

    def max_step(self, r, a_adversary, a_learner, d_adversary: np.array, d_learner: np.array, b_cost):
        t = 1
        while True:
            epsilon_weight = self.epsilon*np.linalg.norm(np.multiply(t, np.concatenate(([d_learner], [d_adversary]))))**2
            upper_bound = b_cost - epsilon_weight
            test = t*d_adversary
            cost = self.arg_max_costs(r, a_adversary + t*d_adversary, a_learner + t*d_learner)[0]
            if cost <= upper_bound:
                break
            else:
                t /= 2
        return t

    # this takes a really long time for large data sets
    def find_min_b(self):
        x0 = [0] * self.num_features*2
        b = optimize.minimize(self.minimize_b, x0)
        return b

    def minimize_b(self, x0: List):
        a_adversary = np.asarray(x0[0:self.num_features])
        a_learner = np.asarray(x0[self.num_features:self.num_features*2])
        return np.dot(self.mu_learner(a_adversary, a_learner),
                      self.mu_adversary(a_adversary, a_learner))

    def psi(self, x0, *args):
        (r, a_adversary, a_learner) = args
        b_adversary = np.asarray(x0[0:self.num_features])
        b_learner = np.asarray(x0[self.num_features:self.num_features*2])

        sum_ = 0.0
        for i in range(0, self.num_features):
            sum_ += r*(self.theta_learner(a_adversary, a_learner)
                       - self.theta_learner(a_adversary, b_learner))

        for i in range(0, self.num_features):
            sum_ += (1-r)*(self.theta_adversary(a_adversary, a_learner)
                       - self.theta_adversary(b_adversary, a_learner))
        return -sum_

    def neq_deriv_loss(self, z_: float, y_):
        return partial_derivative(self.loss_function, var=0, point=[z_, y_])

    def neq_deriv_learner_loss(self, z_: float, y_):
        return partial_derivative(self.learner_loss_function, var=0, point=[z_, y_])

    def mu_adversary(self, a_adversary, a_learner):
        mu = []
        for instance in self.train_instances:
            if instance.get_label() == 1:
                fv = instance.get_feature_vector().get_csr_matrix().toarray()[0]
                new_fv = transform_instance(fv, a_adversary)
                prediction = predict_instance(new_fv, a_learner)
                mu.append(self.neq_deriv_loss(prediction, instance.get_label()))
        return np.asarray(mu)

    def mu_learner(self, a_adversary, a_learner):
        mu = []
        for instance in self.train_instances:
            if instance.get_label() == 1:
                fv = instance.get_feature_vector().get_csr_matrix().toarray()[0]
                new_fv = transform_instance(fv, a_adversary)
                prediction = predict_instance(new_fv, a_learner)
                mu.append(self.neq_deriv_learner_loss(prediction, instance.get_label()))
        return np.asarray(mu)

    def theta_adversary(self, a_adversary, a_learner):
        sum_ = 0
        for instance in self.train_instances:
            if instance.get_label() == 1:
                fv = instance.get_feature_vector().get_csr_matrix().toarray()[0]
                new_fv = transform_instance(fv, a_adversary)
                prediction = predict_instance(new_fv, a_learner)
                sum_ += (self.loss_function(prediction, instance.get_label()) + self.regularizer(a_adversary))
        return sum_

    def theta_learner(self, a_adversary, a_learner):
        sum_ = 0
        for instance in self.train_instances:
            if instance.get_label() == 1:
                fv = instance.get_feature_vector().get_csr_matrix().toarray()[0]
                new_fv = transform_instance(fv, a_adversary)
                prediction = predict_instance(new_fv, a_learner)
                sum_ += (self.learner_loss_function(prediction, instance.get_label()) + self.learner_regularizer(a_learner))
        return sum_


class AntagonisticLossEquilibrium(Game):

    def find_params(self):
        a_adversary = np.zeros(self.num_features)
        a_learner = np.zeros(self.num_features)

        while True:
            a_adversary = self.get_max_a_adversary(a_learner)
            d = np.multiply(-1, self.theta_a_learner(a_adversary, a_learner))
            t = self.max_step(a_adversary, a_learner, d)

            new_a_learner = a_learner + np.multiply(t, d)
            difference = np.linalg.norm(np.subtract(new_a_learner, a_learner))
            a_learner = deepcopy(new_a_learner)

            if difference <= self.epsilon:
                break

        return {'adversary': a_adversary,
                'learner': a_learner}

    def get_max_a_adversary(self, a_learner):
        x0 = [0] * self.num_features
        res = optimize.fmin_slsqp(self.max_min_a, x0, args=(a_learner, 1), full_output=True)
        a_adversary = res[0][0:self.num_features]
        return a_adversary

    def max_min_a(self, x0, *args):
        a_learner = args[0]
        a_adversary = np.asarray(x0[0:self.num_features])
        return -1*self.theta(a_adversary, a_learner)

    def theta(self, a_adversary, a_learner):
        sum_ = 0
        for instance in self.train_instances:
            if instance.get_label() == 1:
                fv = instance.get_feature_vector().get_csr_matrix().toarray()[0]
                new_fv = transform_instance(fv, a_adversary)
                prediction = predict_instance(new_fv, a_learner)
                sum_ += (self.learner_loss_function(prediction, instance.get_label())
                         + self.learner_regularizer(a_learner) - self.regularizer(a_adversary))
        return sum_

    def theta_diff(self, a_adversary, a_learner: np.array, x, index):
        a_learner.itemset(index, x)
        return self.theta(a_adversary, a_learner)

    def theta_a_learner(self, a_adversary: np.array, a_learner: np.array):
        gradient = []
        for i in range(0, self.num_features):
            x = a_learner.item(i)
            diff_ = partial_derivative(self.theta_diff, var=2, point=[a_adversary, a_learner, x, i])
            gradient.append(diff_)
        return np.asarray(gradient)

    def max_step(self, a_adversary, a_learner, d: np.array):
        t = 1
        while True:
            epsilon_weight = self.epsilon*np.linalg.norm(np.multiply(t, d))**2
            upper_bound = self.theta(a_adversary, a_learner) - epsilon_weight
            cost = self.theta(a_adversary, np.add(a_learner, np.multiply(t, d)))
            if cost <= upper_bound:
                break
            else:
                t /= 2
        return t
