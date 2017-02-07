from data_reader.input import Instance, FeatureVector
from adversaries.adversary import Adversary
from learners.learner import InitialPredictor
from typing import List, Dict
from types import FunctionType
import numpy as np
import learners
from scipy import optimize
from copy import deepcopy

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
    def __init__(self, learner_loss_function, adversary_costs, learner_costs, feature_mapping,
                 density_weight, learner_density_weight, max_iterations):
        self.weight_vector = None
        self.tau_factors = None

        self.max_iterations = max_iterations
        self.learner_loss_function = learner_loss_function  # type: FunctionType
        self.adversary_costs = adversary_costs  # type: List[]
        self.learner_costs = learner_costs  # type: List[]
        self.feature_mapping = feature_mapping  # type: FunctionType
        self.density_weight = density_weight  # type: float
        self.learner_density_weight = learner_density_weight  # type: float

    def optimize(self, train_instances, num_features):
        raise NotImplementedError

    def learner_optimization(self, x0: List, *args):
        raise NotImplementedError

    def constraint(self, x0: List, *args):
        raise NotImplementedError

    def loss_function(self, z: float, y: int):
        raise NotImplementedError


class LinearLoss(SPG):
    def optimize(self, train_instances, num_features):
        n = len(train_instances)
        self.tau_factors = [-n / self.density_weight * cost for cost in self.adversary_costs]
        x0 = [2.0] * (len(train_instances) + num_features)
        labels = [x.get_label() for x in train_instances]
        feature_mappings = [self.feature_mapping(x.get_feature_vector()) for x in train_instances]
        result = optimize.fmin_slsqp(self.learner_optimization, x0,
                                     args=(n, num_features, feature_mappings, labels),
                                     f_ieqcons=self.constraint, iter=self.max_iterations)
        self.weight_vector = result[0:num_features]
        return {'tau_factors': self.tau_factors, 'weight_vector': self.weight_vector}

    def learner_optimization(self, x0: List, *args):
        num_instances = args[0]
        num_features = args[1]

        weight_vector_ = np.asarray(x0[0:num_features])
        norm_ = np.linalg.norm(weight_vector_)
        xi_factors_ = x0[num_features:num_instances + num_features]

        constant_ = (self.learner_density_weight / 2) * norm_

        sum_ = 0
        for i in range(0, num_instances):
            sum_ += self.learner_costs[i] * xi_factors_[i]

        return sum_ + constant_

    def constraint(self, x0: List, *args):
        (num_instances, num_features, feature_mappings, labels) = args
        weight_vector_ = np.asarray(x0[0:num_features])
        norm_ = np.linalg.norm(weight_vector_)
        xi_factors_ = x0[num_features:num_instances + num_features]
        cons = [labels[i] * np.dot(weight_vector_, feature_mappings[i]) - 1
                - (labels[i] * -self.tau_factors[i] * norm_) + xi_factors_[i] for i in range(0, num_instances)]

        return cons + xi_factors_

    def loss_function(self, z: float, y: int):
        return z



class WorstCaseLoss(SPG):
    '''
        In the case of worst-case loss, learner and attacker
         loss function cannot be both convex, so it does not have
         a meaningful tau-factor.

    '''
    def optimize(self, train_instances, num_features):
        # TODO fix lower level optimization

        self.tau_factors = [1]*(len(train_instances) - num_features)
        x0 = [0.0] * (len(train_instances) + num_features)
        labels = [x.get_label() for x in train_instances]
        feature_mappings = [self.feature_mapping(x.get_feature_vector()) for x in train_instances]
        result = optimize.fmin_slsqp(self.learner_optimization, x0,
                                     args=(len(train_instances), num_features, feature_mappings, labels),
                                     f_ieqcons=self.constraint, iter=self.max_iterations)

    def learner_optimization(self, x0: List, *args):
        num_instances = args[0]
        num_features = args[1]

        weight_vector_ = np.asarray(x0[0:num_features])
        norm_ = np.linalg.norm(weight_vector_)
        xi_factors_ = x0[num_features:num_instances + num_features]
        constant_ = (self.learner_density_weight / 2) * norm_
        sum_ = 0
        for i in range(0, num_instances):
            sum_ += self.learner_costs[i] * xi_factors_[i]

        return sum_ + constant_


    def constraint(self, x0: List, *args):
        (num_instances, num_features, feature_mappings, labels) = args
        weight_vector_ = np.asarray(x0[0:num_features])
        norm_ = np.linalg.norm(weight_vector_)
        xi_factors_ = x0[num_features:num_instances + num_features]

        return xi_factors_ - self.lower_level_optimize(x0[0:num_instances],labels)

    def lower_level_optimize(self, instances: List, labels):
        # TODO: fix minimize input
        return optimize.minimize(self.lower_level_func, instances, args=(labels))

    def lower_level_func(self, x, label):
        return (-1) * self.learner_loss_function(self.feature_mapping(x), label);

    def loss_function(self, z: float, y: int):
        # not used in optimization
        return -1 * self.learner_loss_function(z, y)


class LogisticLoss(SPG):
    def optimize(self, train_instances, num_features):
        x0 = [0.0] * (len(train_instances) + num_features)
        labels = [x.get_label() for x in train_instances]
        feature_mappings = [self.feature_mapping(x.get_feature_vector()) for x in train_instances]
        result = optimize.fmin_slsqp(self.learner_optimization, x0,
                                     args=(len(train_instances), num_features, feature_mappings, labels),
                                     f_ieqcons=self.constraint, iter=self.max_iterations)
        self.weight_vector = result[0:num_features]
        self.tau_factors = result[num_features:len(train_instances) + num_features]
        self.tau_factors = self.tau_factors.tolist()
        return {'tau_factors': self.tau_factors, 'weight_vector': self.weight_vector}

    def learner_optimization(self, x0: List, *args):
        (num_instances, num_features, feature_mappings, labels) = args

        weight_vector_ = np.asarray(x0[0:num_features])
        norm_ = np.linalg.norm(weight_vector_)
        tau_factors_ = x0[num_features:num_instances + num_features]

        constant_ = (self.learner_density_weight / 2) * norm_

        sum_ = 0
        for i in range(0, num_instances):
            sum_ += self.learner_costs[i] * \
                    self.learner_loss_function(np.dot(weight_vector_, feature_mappings[i]) +
                                               tau_factors_[i] * norm_, labels[i])
        return sum_ + constant_

    def constraint(self, x0: List, *args):
        (num_instances, num_features, feature_mappings, labels) = args

        weight_vector = np.asarray(x0[0:num_features])
        norm_ = np.linalg.norm(weight_vector)
        tau_factors = x0[num_features:num_instances + num_features]
        cons = [tau_factors[i] * (1 + np.exp(-1 * np.dot(weight_vector, feature_mappings[i]) - tau_factors[i] * norm_))
                + num_instances / self.density_weight * self.adversary_costs[i] for i in range(0, num_instances)]

        return cons

    def loss_function(self, z: float, y: int):
        return np.log(1 + np.exp(z))


class Stackelberg(Adversary):
    LINEAR_LOSS = 'linear_loss'
    WORST_CASE_LOSS = 'worst_case_loss'
    LOGISTIC_LOSS = 'logistic_loss'

    def __init__(self):

        self.game = Adversary.LINEAR_LOSS
        self.max_iterations = 10

        self.train_instances = None  # type: List[Instance]
        self.num_features = None  # type: int
        self.feature_mapping = None  # type: FunctionType

        self.adversary_costs = None  # type: List[]
        self.learner_costs = None
        self.density_weight = 0.1  # type: float

        self.tau_factors = None  # type: List[]
        self.weight_vector = None  # type: np.array

    def attack(self, instances: List[Instance]) -> List[Instance]:
        transformed_instances = []
        current_instance = 0
        for instance in self.train_instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == 1:
                transformed_instances.append(self.transform(transformed_instance, current_instance))
            else:
                transformed_instances.append(transformed_instance)
            current_instance += 1
        return transformed_instances

    def transform(self, instance: Instance, index: int):
        new_instance = np.add(np.multiply(self.tau_factors[index], self.weight_vector),
                              self.feature_mapping(instance.get_feature_vector()))
        new_instance = new_instance.tolist()
        features = []
        for j in range(0, len(new_instance)):
            if new_instance[j] >= 0.5:
                features.append(j)
        transformed_instance = Instance(instance.get_label(), FeatureVector(self.num_features, features))
        return transformed_instance

    def set_params(self, params: Dict):
        if params['game'] is not None:
            self.game = params['game']

        if params['adversary_costs'] is not None:
            self.adversary_costs = params['adversary_costs']

        if params['learner_costs'] is not None:
            self.adversary_costs = params['adversary_costs']

        if params['density_weight'] is not None:
            self.density_weight = params['density_weight']

    def get_available_params(self) -> Dict:
        params = {'game': self.game,
                  'adversary_costs': self.adversary_costs,
                  'learner_costs': self.learner_costs,
                  'density_weight': self.density_weight}
        return params

    def set_adversarial_params(self, learner, train_instances):
        learner_loss_function = learner.get_loss_function()
        learner_density_weight = learner.get_density_weight()

        self.train_instances = train_instances  # type: List[Instance]
        self.num_features = self.train_instances[0].get_feature_vector().get_feature_count()
        self.feature_mapping = learner.get_feature_mapping()
        if self.adversary_costs is None:
            self.adversary_costs = [1.0 / len(self.train_instances)] * len(self.train_instances)
        if self.learner_costs is None:
            self.learner_costs = [1.0 / len(self.train_instances)] * len(self.train_instances)

        self.train_instances = train_instances
        self.num_features = self.train_instances[0].get_feature_vector().get_feature_count()

        if self.game is Adversary.LINEAR_LOSS:
            solution = LinearLoss(learner_loss_function, self.adversary_costs, self.learner_costs, self.feature_mapping,
                                  self.density_weight, learner_density_weight, self.max_iterations)
            transforms = solution.optimize(self.train_instances, self.num_features)
            self.tau_factors = transforms['tau_factors']
            self.weight_vector = transforms['weight_vector']


        elif self.game is Adversary.WORST_CASE_LOSS:
            solution = WorstCaseLoss(learner_loss_function, self.adversary_costs, self.learner_costs,
                                     self.feature_mapping,
                                     self.density_weight, learner_density_weight, self.max_iterations)
            transforms = solution.optimize(self.train_instances, self.num_features)
            self.perturbation_vector = transforms['adversary']
            self.learner_perturbation_vector = transforms['learner']

        elif self.game is Adversary.LOGISTIC_LOSS:
            solution = LogisticLoss(learner_loss_function, self.adversary_costs, self.learner_costs,
                                    self.feature_mapping,
                                    self.density_weight, learner_density_weight, self.max_iterations)
            transforms = solution.optimize(self.train_instances, self.num_features)
            self.tau_factors = transforms['tau_factors']
            self.weight_vector = transforms['weight_vector']

        else:
            raise ValueError('unspecified stackelberg game')

    def get_learner_params(self):
        return self.weight_vector
