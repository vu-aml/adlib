from adversaries.adversary import Adversary
from data_reader.binary_input import Instance, BinaryFeatureVector
from typing import List, Dict
from random import shuffle
import numpy as np
from copy import deepcopy
from learners import SimpleLearner
from sklearn.svm import SVC


# import matplotlib.pyplot as plt



class BinaryGreedy(Adversary):
    def __init__(self, learner=None, max_change=200,
                 lambda_val=0.05, epsilon=0.00000001, step_size=0.05, cost_function = "quadratic"):
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
        self.cost_function = cost_function
        self.max_change = max_change
        if self.learn_model is not None:
            self.weight_vector = self.learn_model.get_weight()
        else:
            self.weight_vector = None  # type: np.array

    def get_available_params(self) -> Dict:
        return {"lambda_val":self.lambda_val,
                "epsilon":self.epsilon,
                "step_size": self.step_size,
                "max_change":self.max_change,
                "cost_function":self.cost
                }

    def set_params(self, params: Dict):
        if "lambda_val" in params.keys():
            self.lambda_val = params["lambda_val"]
        if "epsilon" in params.keys():
            self.epsilon = params["epsilon"]
        if "step_size" in params.keys():
            self.step_size = params["step_size"]
        if "max_change" in params.keys():
            self.max_change = params["max_change"]
        if "cost_function" in params.keys():
            self.cost_function = params["cost_function"]

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
                transformed_instances.append(self.coordinate_greedy(instance))
            else:
                transformed_instances.append(instance)
        return transformed_instances

    def coordinate_greedy(self, instance: Instance) -> Instance:
        indices = [i for i in range(0, self.num_features)]

        x = xk = instance.get_csr_matrix().toarray()[0]
        # Q = [self.transform_cost(xk,x)]
        # f = [self.learn_model.model.learner.predict(xk.reshape(1,-1))]
        # p = [self.learn_model.model.learner.coef_.dot(xk)+self.learn_model.model.learner.intercept_]
        # c = [self.quadratic_cost(xk,x)]

        no_improve_count = 0
        shuffle(indices)
        for i in indices:
            xkplus1 = self.minimize_transform(xk, i)
            oldQ = self.transform_cost(xk, x)
            newQ = self.transform_cost(xkplus1, x)
            # step_change = np.log(newQ) / np.log(oldQ)
            # using difference instead of log ratio for convergence check

            step_change = newQ - oldQ
            # print('oldQ= '+str(oldQ) + ' newQ= '+str(newQ)+ ' step_change= '+str(step_change))
            # print('xk[i]= ' + str(xk[i]) + ' xk+1[i]= ' + str(xkplus1[i]) + ' x[i]= ' + str(x[i]))

            if step_change >= 0:
                no_improve_count += 1
                if no_improve_count >= self.max_change:
                    break
            else:
                xk = xkplus1

                # Q.append(self.transform_cost(xk,x))
                # f.append(self.learn_model.model.learner.predict(xk.reshape(1, -1)))
                # c.append(self.quadratic_cost(xk,x))
                # p.append(self.learn_model.model.learner.coef_.dot(xk) + self.learn_model.model.learner.intercept_)

        # print('xk shape: '+str(xk.shape))

        # Q = np.array(Q)
        # f = np.array(f)
        # c = np.array(c)
        # p = np.array(p).reshape((-1,))
        # pnc = p+c
        # print(p.shape)
        # print(c.shape)
        # print(pnc.shape)
        # t = np.array([i for i in range(len(Q))])
        # plt.plot(t,Q,'r', label='Q(x)')
        # plt.plot(t, f, 'b', label='sign(f(x))')
        # plt.plot( t,c ,'g', label='||x-xi||^2')
        # plt.plot(t, p, 'b--',label='w.T*x+b')
        # plt.plot(t, pnc, 'r--',
        #          label='w.T*x+b + ||x-xi||^2')
        # plt.legend()
        # plt.show()

        # ('mod succeeded')

        mat_indices = [x for x in range(0, self.num_features) if xk[x] != 0]
        new_instance = Instance(-1, BinaryFeatureVector(self.num_features, mat_indices))

        if self.learn_model.predict(new_instance) == self.learn_model.positive_classification:
            return instance
        else:
            return new_instance

    def minimize_transform(self, xi: np.array, i):
        xk = np.copy(xi)
        xk[i] = 1 - xi[i]
        return xk

    def transform_cost(self, x: np.array, xi: np.array):
        if self.cost_function == "quadratic":
            return self.learner_predict(x) + self.quadratic_cost(x, xi)
        elif self.cost_function == "exponential":
            return self.learner_predict(x) + self.exponential_cost(x,xi)

    def quadratic_cost(self, x: np.array, xi: np.array):
        return self.lambda_val / 2 * sum((x - xi) ** 2)


    def exponential_cost(self, x:np.array, xi: np.array):
        return np.exp(self.lambda_val * np.sqrt(np.sum((x - xi) **2) + 1))


    def learner_predict(self,attack_instance):
        if self.learn_model == SimpleLearner and self.learn_model.model.learner == SVC:
            param_map = self.learn_model.get_params()
            attribute_map = self.learn_model.get_attributes()
            if param_map["kernel"] == "rbf":
               return self.learn_model.model.learner.decision_function(attack_instance.reshape(1,-1))
            if param_map["kernel"] == "linear":
                return attribute_map["coef_"][0].dot(attack_instance) + attribute_map["intercept_"]
        else:
            return self.learn_model.get_weight().dot(attack_instance) + self.learn_model.get_constant()
