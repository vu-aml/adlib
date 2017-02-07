import pickle
from adversaries.adversary import Adversary
from data_reader.input import Instance, FeatureVector
from typing import List, Dict
from random import shuffle
import numpy as np
from scipy import optimize
from copy import deepcopy

'''Companion adversary alg to Retraining improved learner.

Concept:
    Randomly iterates through features in an adversarial instance, to greedily find
    lowest cost (optimal transform).
'''

class CoordinateGreedy(Adversary):

    def __init__(self, lambda_val=1.0, epsilon=0.1, learner=None):
        self.lambda_val = lambda_val
        self.epsilon = epsilon
        self.num_features = None
        self.learner = learner         #type: Classifier


    def attack(self, instances) -> List[Instance]:
        transformed_instances = []
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == 1:
                transformed_instances.append(self.coordinate_greedy(transformed_instance, instances))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def set_params(self, params: Dict):
        if params['lambda_val'] is not None:
            self.lambda_val = params['lambda_val']

        if params['epsilon'] is not None:
            self.epsilon = params['epsilon']

    def get_available_params(self) -> Dict:
        params = {'lambda_val': self.lambda_val,
                  'epsilon': self.epsilon,}
        return params

    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learner = learner
        self.num_features = train_instances[0].get_feature_vector().get_feature_count()

    def coordinate_greedy(self, instance: Instance, instances: List[Instance]):
        x = [i for i in range(0, self.num_features)]
        initial = instance.get_feature_vector().get_csr_matrix().toarray()[0]
        final = instance.get_feature_vector().get_csr_matrix().toarray()[0]

        suboptimal = True
        while suboptimal:
            x = shuffle(x)
            for i in x:
                features = instances[i].get_feature_vector().get_csr_matrix().toarray()[0]
                new_features = self.minimize_transform(features)
                step = np.log(self.transform_cost(new_features, features)) / np.log(self.transform_cost(final, features))
                final = deepcopy(new_features)
                if step <= self.epsilon:
                    suboptimal = False
                    break

        if self.learner.decision_function([final]) >= 0:
            final = deepcopy(initial)

        indices = [x for x in range(0,self.num_features) if final[x] == 1]
        return Instance(1, FeatureVector(self.num_features, indices))

    def minimize_transform(self, xi: np.array):
        x0 = np.zeros(self.num_features)
        optimize.minimize(self.transform_cost, x0, args=xi)
        return x0

    def transform_cost(self, x: np.array, xi: np.array):
        return self.learner.decision_function([x])[0] + self.quadratic_cost(x, xi)

    def quadratic_cost(self, x: np.array, xi: np.array):
        cost = self.lambda_val/2 * np.linalg.norm(np.subtract(x, xi))**2
        return cost
