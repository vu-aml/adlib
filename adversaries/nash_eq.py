from adversaries.adversary import Adversary
from adversaries.nash_eq_games import ConvexLossEquilibirum, AntagonisticLossEquilibrium
from util import transform_instance, partial_derivative, predict_instance
from data_reader.input import Instance, FeatureVector
from typing import List, Dict
from types import FunctionType
import numpy as np
from scipy import optimize
from copy import deepcopy

'''Nash equilibrium solution between adversary and improved learner.

Concept:
    Given loss functions for the learner and adversary and the costs for each
    instance in the training set, find the pair of vectors (w+, w-) such that the
    adversary transforms the data over w+ and the learner finds a decision value
    for each instance using w-. Generates the pair (w+, w-) such that learner and
    adversary reach their nash equilibrium strategy sets; neither can benefit by
    straying from the pairing.

    Solution is generated in one round (see stackelberg for multi-round).

'''

class NashEq(Adversary):

    CONVEX_LOSS = 'convex_loss'
    ANTAGONISTIC_LOSS = 'antagonistic_loss'

    def __init__(self):
        self.game = NashEq.CONVEX_LOSS                         # type: str
        self.learner_loss_function = None                      # type: FunctionType
        self.loss_function_name = 'neq_linear_loss'            # type: FunctionType
        self.train_instances = None                            # type: List[Instance]
        self.num_features = None                               # type: int
        self.lambda_val = 1.0                                  # type: float
        self.learner_lambda_val = 1.0                          # type: float
        self.epsilon = 1.0                                     # type: float

        self.perturbation_vector = None                        # type: np.array
        self.learner_perturbation_vector = None                # type: np.array

    def attack(self, instances) -> List[Instance]:
        transformed_instances = []
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == 1:
                transformed_instances.append(self.neq_solution(transformed_instance))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def neq_solution(self, instance: Instance):
        fv = instance.get_feature_vector().get_csr_matrix().toarray()
        new_instance = transform_instance(fv, self.perturbation_vector)[0]
        features = []
        for j in range(0, self.num_features):
            if new_instance[j] >= 0.5:
                features.append(j)
        transformed_instance = Instance(1, FeatureVector(self.num_features, features))
        return transformed_instance

    def set_params(self, params: Dict):
        if params['game'] is not None:
            self.game = params['game']

        if params['loss_function_name'] is not None:
            self.loss_function_name = params['loss_function_name']

        if params['lambda_val'] is not None:
            self.lambda_val = params['lambda_val']

        if params['epsilon'] is not None:
            self.epsilon = params['epsilon']

    def get_available_params(self) -> Dict:
        params = {'game': self.game,
                  'loss_function_name': self.loss_function_name,
                  'lambda_val': self.lambda_val,
                  'epsilon': self.epsilon}
        return params

    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learner_loss_function = learner.get_loss_function()
        self.train_instances = train_instances
        self.num_features = self.train_instances[0].get_feature_vector().get_feature_count()

        if self.game == NashEq.CONVEX_LOSS:
            solution = ConvexLossEquilibirum(self.loss_function(), self.learner_loss_function, self.train_instances,
                                             self.num_features, self.epsilon, self.lambda_val, self.learner_lambda_val)
            # find_params takes a really long time to execute, possibly due to the "while True" loop
            transforms = solution.find_params()
            self.perturbation_vector = transforms['adversary']
            self.learner_perturbation_vector = transforms['learner']

        elif self.game == NashEq.ANTAGONISTIC_LOSS:
            solution = AntagonisticLossEquilibrium(self.loss_function(), self.learner_loss_function, self.train_instances,
                                             self.num_features, self.epsilon, self.lambda_val, self.learner_lambda_val)
            transforms = solution.find_params()
            self.perturbation_vector = transforms['adversary']
            self.learner_perturbation_vector = transforms['learner']
        else:
            raise ValueError('unspecified equilibrium algorithm')

    def loss_function(self):
        return getattr(self, self.loss_function_name)

    def neq_worst_case_loss(self, z: float, y: int) -> float:
        return -1 * self.learner_loss_function(z,y)

    def neq_linear_loss(self, z: float, y: int) -> float:
        return z

    def neq_logistic_loss(self, z: float, y: int) -> float:
        return np.log(1 + np.exp(z))

    def get_learner_params(self):
        return self.learner_perturbation_vector
