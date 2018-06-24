from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from data_reader.real_input import RealFeatureVector
from typing import List, Dict
import numpy as np
from adlib.learners import Learner

#import matplotlib.pyplot as plt
#for debugging purposes, the Q values are printed
from random import *

DEBUG = False

"""
   Based on A General Retraining Framework for Scalable Adversarial Classification
   paper by Bo Li, Yevgeniy Vorobeychik and Xinyun Chen.

   Concept: the attacker changes the training data instances by iteratively choose a
            feature, and greedily update this feature by minimizing transform cost.
            Update the list of training data.
"""


class CoordinateGreedy(Adversary):
    def __init__(self, learn_model=None, max_iteration=1000,
                 lambda_val=0.01, epsilon=1e-9, step_size=1, cost_function = "quadratic",random_start = 3,
                 convergence_time = 100):
        """
        :param learner: Learner(from learners)
        :param max_iteration: max times allowed to change the feature
        :param lambda_val: weight in quodratic distances calculation
        :param epsilon: the limit of difference between transform costs of ,xij+1, xij, and orginal x
        :param step_size: weight for coordinate descent
        :param cost_function: decide whether to use exponential cost or quadratic cost
        """
        Adversary.__init__(self)
        self.lambda_val = lambda_val
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_features = 0
        self.bias = 0
        self.learn_model = learn_model
        self.max_iteration = max_iteration
        self.cost_function = cost_function
        self.random_start = random_start
        self.convergence_time = convergence_time
        if self.learn_model is not None:
            self.weight_vector = self.learn_model.get_weight()
        else:
            self.weight_vector = None  # type: np.array

    def get_available_params(self) -> Dict:
        return {'max_iteration':self.max_iteration,
                'lambda_val':self.lambda_val,
                'epsilon':self.epsilon,
                'step_size':self.step_size,
                'random_start':self.random_start,
                'cost_function':self.cost_function,
                'convergence_time':self.convergence_time}

    def set_params(self, params: Dict):
        if 'max_iteration' in params.keys():
            self.max_iteration = params['max_iteration']
        if 'lambda_val' in params.keys():
            self.lambda_val = params['lambda_val']
        if 'epsilon' in params.keys():
            self.f_attepsilonack = params['epsilon']
        if 'step_size' in params.keys():
            self.step_size = params['step_size']
        if 'random_start' in params.keys():
            self.random_start = params['random_start']
        if 'cost_function' in params.keys():
            self.cost_function = params['cost_function']
        if 'convergence_time' in params.keys():
            self.convergence_time = params['convergence_time']


    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_count()

    def attack(self, Instances) -> List[Instance]:
        if self.weight_vector is None and self.learn_model is not None:
            self.weight_vector = self.learn_model.get_weight()
            self.bias = self.learn_model.get_constant()
        if self.num_features == 0:
            self.num_features = Instances[0].get_feature_count()

        if self.weight_vector is None:
            raise ValueError('Must set learner_model and weight_vector before attack.')
        print("weight vec before attack: {}".format(self.weight_vector.shape))

        transformed_instances = []
        for instance in Instances:
            if instance.label > 0:
                 #newx = self.coordinate_greedy(instance).get_csr_matrix().toarray().tolist()[0]
                 #oldx = instance.get_csr_matrix().toarray().tolist()[0]
                 # for i in range(len(oldx)):
                 #     if abs(newx[i]-oldx[i]) > 0.005:
                 #         print("index {} has changed by {}".format(i,abs(newx[i]-oldx[i]) ))
                 transformed_instances.append(self.random_start_coordinate_greedy(instance))
            else:
              transformed_instances.append(instance)

        #if DEBUG:
        #    plt.show()

        return transformed_instances

    def coordinate_greedy(self, instance: Instance):
        """
         Greedily update the feature to incrementally improve the attackers utility.
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
        instance_len = instance.get_feature_count()
        if DEBUG:
            iteration_list = []
            Q_value_list = []

        x = xk = instance.get_csr_matrix().toarray()[0]

        # converge is used for checking convergance conditions
        # if the last convergence_time iterations all satisfy <= eplison condition
        # ,the attack successfully finds a optimum
        converge = 0

        for iteration_time in range(self.max_iteration):
            i = randint(0,instance_len - 1)

            #calcualte cost function and greediy improve from a random feature i
            xkplus1 = self.minimize_transform(xk, x, i)
            old_q = self.transform_cost(xk, x)
            new_q = self.transform_cost(xkplus1, x)

            # check whether Q_value actually descends and converges to a minimum
            # plot the iteration and Q_values using matplotlib
            #if DEBUG:
            #    iteration_list.append(iteration_time)
            #    Q_value_list.append(new_q)

           # if new_q < 0:
           #     print("Attack finishes because Q is less than 0")
           #     break

            if new_q - old_q <= 0:
                xk = xkplus1
                step_change = old_q - new_q
                # the np.log() may not converge in special cases
                # makes sure the cost function actually converges
                # alternative implementation?
                #step_change = np.log(new_q) / np.log(old_q)
                #step_change = np.log(old_q - new_q)

                if step_change <= self.epsilon:
                    converge += 1
                    if converge >= self.convergence_time:
                        #print("Attack finishes because of convergence!")
                        break

        #if DEBUG:
        #    plt.plot(iteration_list,Q_value_list)

        mat_indices = [x for x in range(0, self.num_features) if xk[x] != 0]
        mat_data = [xk[x] for x in range(0, self.num_features) if xk[x] != 0]
        new_instance = Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))
        return new_instance


    def random_start_coordinate_greedy(self,instance: Instance):
        """
        implement the n random start algorithm by performing CG for n times.
        The minimized Q and x is used as new attack instance.
        :param instance:
        :return:
        """
        instance_lst = []
        q_value_lst = []
        old_x = instance.get_csr_matrix().toarray()[0]
        for i in range(self.random_start):
            new_attacked_instance = self.coordinate_greedy(instance)
            x = new_attacked_instance.get_csr_matrix().toarray()[0]
            q = self.transform_cost(x, old_x)
            instance_lst.append(new_attacked_instance)
            q_value_lst.append(q)
        return min(zip(instance_lst,q_value_lst),key=lambda x:x[1])[0]



    def minimize_transform(self, xi: np.array, x: np.array, i):
        xk = np.copy(xi)
        # print(xk.shape)
        # print("the index is {}".format(i))
        # print("shape of weight vector is {}".format(self.weight_vector.shape))
        #print("the weight_vector[i] is {}".format(self.weight_vector[i]))
        #print("the xk[i] is {}".format(xk[i]))
        #print("the x[i] is {}".format(x[i]))

        val = 0
        if self.cost_function == "quadratic":
            #ADD val checking to make sure updated value is greater than 0
            #for email data, feature < 0 does not make sense
            val = self.step_size * (self.weight_vector[i] + self.lambda_val * (xk[i] - x[i]))
        elif self.cost_function == "exponential":
            val = self.step_size * (self.weight_vector[i] +
                                    self.lambda_val * self.exponential_cost(x, xi) *
                                    (1 / np.sqrt(np.sum((x - xi) **2) + 1)) * (xk[i] - x[i]))
        #print("the weight_vector[i] is {}".format(self.weight_vector[i]))
        #print("the x[i] is {}".format(x[i]))
        #print("the old xk[i] is {}".format(xk[i]))
        if xk[i] - val >= 0:
            xk[i] -= val
        #print(val)
        ##print("the new xk[i] is {}".format(xk[i]))
        return xk


    def transform_cost(self, x: np.array, xi: np.array):
        if self.cost_function == "quadratic":
            return self.weight_vector.dot(x) + self.bias + self.quadratic_cost(x, xi)
        elif self.cost_function == "exponential":
            return self.weight_vector.dot(x) + self.bias + self.exponential_cost(x,xi)


    def quadratic_cost(self, x: np.array, xi: np.array):
        return self.lambda_val / 2 * sum((x - xi) ** 2)


    def exponential_cost(self, x:np.array, xi: np.array):
        return np.exp(self.lambda_val * np.sqrt(np.sum((x - xi) **2) + 1))