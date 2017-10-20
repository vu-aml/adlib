from adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from data_reader.real_input import RealFeatureVector
from typing import List, Dict
from random import shuffle
import numpy as np
import learners as learners
from copy import deepcopy

"""
   Based on A General Retraning Framwork for Scalable Adversarial Classification
   paper by Bo Li, Yevgeniy Vorobeychik and Xinyun Chen.

   Concept: the attacker change the training data instances by iteratively choose a
            feature, and greedily update this feature by minimizing transform cost.
            Update the list of training data.
"""


class CoordinateGreedy(Adversary):
    def __init__(self, learner=None, max_change=100,
                 lambda_val=0.05, epsilon=0.0002, step_size=1):
        """
        :param learner: Learner(from learners)
        :param max_change: max times allowed to change the feature
        :param lambda_val: weight in quodratic distances calculation
        :param epsilon: the limit of difference between transform costs of ,xij+1, xij, and orginal x
        :param step_size: weight for coordinate descent
        """
        Adversary.__init__(self)
        self.lambda_val = lambda_val
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_features = 0
        self.learn_model = learner
        self.max_change = max_change
        if self.learn_model is not None:
            self.weight_vector = self.learn_model.get_weight()
        else:
            self.weight_vector = None  # type: np.array

    def get_available_params(self) -> Dict:
        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_count()

    def attack(self, Instances) -> List[Instance]:
        if self.weight_vector is None and self.learn_model is not None:
            self.weight_vector = self.learn_model.get_weight()
        if self.num_features == 0:
            self.num_features = Instances[0].get_feature_count()

        if self.weight_vector is None:
            raise ValueError('Must set learner_model and weight_vector before attack.')

        transformed_instances = []
        for instance in Instances:
            if instance.label > 0:
                 newx = self.coordinate_greedy(instance).get_csr_matrix().toarray().tolist()[0]
                 oldx = instance.get_csr_matrix().toarray().tolist()[0]
                 # for i in range(len(oldx)):
                 #     if abs(newx[i]-oldx[i]) > 0.005:
                 #         print("index {} has changed by {}".format(i,abs(newx[i]-oldx[i]) ))
                 transformed_instances.append(self.coordinate_greedy(instance))
            else:
              transformed_instances.append(instance)
        return transformed_instances

    def coordinate_greedy(self, instance: Instance) -> Instance:
        """
         Greddily update the feature to incrementally improve the attackers utility.
         run CS from L random starting points in the feature space. We repeat the
         alternation until differences of instances are small or max_change is
         reached.

         no_improve_count: number of points
         Q: transofrm cost（we use quodratic distance）
         GreedyImprove: using the coordinate descent algorithm.
        :param instance:
        :return: if the result is still classified as +1, we return origin instance
                 else we return the improved.
        """
        indices = [i for i in range(0, self.num_features)]
        x = xk = instance.get_csr_matrix().toarray()[0]
        no_improve_count = 0
        shuffle(indices)
        count = 0
        for i in indices:

            xkplus1 = self.minimize_transform(xk, x, i)
            oldQ = self.transform_cost(xk, x)
            newQ = self.transform_cost(xkplus1, x)
            # step_change = np.log(newQ) / np.log(oldQ)
            # using difference instead of log ratio for convergence check

            xk = xkplus1
            no_improve_count += 1
            if newQ - oldQ > 0 and oldQ != 0:
                step_change = np.log(newQ - oldQ)
                if step_change <= self.epsilon:
                    break
            if no_improve_count > self.max_change:
                break
            count += 1
        mat_indices = [x for x in range(0, self.num_features) if xk[x] != 0]
        mat_data = [xk[x] for x in range(0, self.num_features) if xk[x] != 0]
        new_instance = Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))

        return new_instance

    def minimize_transform(self, xi: np.array, x: np.array, i):
        xk = np.copy(xi)
        xk[i] -= self.step_size * (self.weight_vector[i] + self.lambda_val * (xk[i] - x[i]))
        return xk

    def transform_cost(self, x: np.array, xi: np.array):
        return self.weight_vector.dot(x) + self.quadratic_cost(x, xi)

    def quadratic_cost(self, x: np.array, xi: np.array):
        return self.lambda_val / 2 * sum((x - xi) ** 2)
