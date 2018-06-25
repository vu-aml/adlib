from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from data_reader.real_input import RealFeatureVector
from adlib.learners.simple_learner import SimpleLearner
from typing import List, Dict
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import pairwise
from data_reader.operations import sparsify
import operator

# import matplotlib.pyplot as plt

"""
    Gradient Desecent Evasion Attack from Evasion Attacks Against Machine Learning at Test Time,
    written by Battista Biggio, Igino Corona, et. al.
    Concept: The gradient descent based attack that modifies the malicious sample by minimizing its
    classification function
    (g(x)) result, subject to a bound on its distance. An extra mimicry component is added to avoid
    descending to meaningless
    regions.

    This algorithm is based on adversariaLib - Advanced library for the evaluation of machine
    learning algorithms and classifiers against adversarial attacks. Copyright (C) 2013, Igino
    Corona, Battista Biggio, Davide Maiorca, Dept. of Electrical and Electronic Engineering,
    University of Cagliari, Italy.
"""


class GradientDescent(Adversary):
    def __init__(self, learn_model=None, step_size=0.01, trade_off=10,
                 stp_constant=0.000000001, mimicry='euclidean',
                 max_iter=1000, mimicry_params={}, bound=0.1):
        """
        :param learner: Learner(from learners)
        :param max_change: max times allowed to change the feature
        :param lambda_val: weight in quodratic distances calculation
        :param epsilon: the limit of difference between transform costs of xij+1,xij, and original x
        :param step_size: weight for coordinate descent
        :param max_boundaries: maximum number of gradient descent iterations to be performed
                               set it to a large number by default.
        :param mimicry_params: hyperparameter for mimicry params.
        :param bound: limit how far one instance can move from its root instance.
                      this is d(x,x_prime) in the algortihm.
        """
        Adversary.__init__(self)
        self.step_size = step_size
        self.lambda_val = trade_off
        self.num_features = 0
        self.learn_model = learn_model
        self.epsilon = stp_constant
        self.mimicry = mimicry
        self.max_iter = max_iter
        self.mimicry_params = mimicry_params
        self.bound = bound

    def get_available_params(self) -> Dict:
        return {'step_size': self.step_size,
                'trade_off': self.lambda_val,
                'learn_model': self.learn_model,
                'stp_constant': self.epsilon,
                'mimicry': self.minicry,
                'max_iteration': self.max_iteration,
                'mimicry_params': self.mimicry_params,
                'bound': self.bound}

    def set_params(self, params: Dict):
        if 'step_size' in params.keys():
            self.step_size = params['step_size']
        if 'trade_off' in params.keys():
            self.lambda_val = params['trade_off']
        if 'stp_constant' in params.keys():
            self.epsilon = params['stp_constant']
        if 'learn_model' in params.keys():
            self.learn_model = params['learn_model']
        if 'self.minicry' in params.keys():
            self.minicry = params['minicry']
        if 'max_iteration' in params.keys():
            self.max_iteration = params['max_iteration']
        if 'mimicry_params' in params.keys():
            self.mimicry_params = params['mimicry_params']
        if 'bound' in params.keys():
            self.bound = params['bound']

    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_count()
        self.train_instances = train_instances

    def attack(self, Instances) -> List[Instance]:
        if self.num_features == 0:
            self.num_features = Instances[0].get_feature_count()
        benign_instances = []
        malicious_instances = []
        for instance in Instances:
            if instance.label < 0:
                benign_instances.append(instance)

        # make negative instances into numpy array for calculate KDE distances
        y_list, X_list = sparsify(benign_instances)
        num_instances = len(y_list)
        y, X = np.array(y_list).reshape((num_instances, 1)), X_list.toarray().reshape(
            (num_instances, self.num_features))

        transformed_instances = []
        for instance in Instances:
            if instance.label < 0:
                transformed_instances.append(instance)
            else:
                transformed_instances.append(self.gradient_descent(instance, X))

        # plt.show()
        return transformed_instances

    def gradient_descent(self, instance: Instance, neg_instances):
        # store iteration and objective values for plotting....
        # iteration_lst = []
        # objective_lst = []

        # attack_intance-> np array
        attack_instance = instance.get_csr_matrix().toarray()
        root_instance = attack_instance
        obj_function_value_list = []

        # store the modified gradient descent attack instances
        # find a list of potential neg_instances, the closest distance, and updated gradients
        candidate_attack_instances = [attack_instance]
        attacker_score = self.get_score(attack_instance)
        closer_neg_instances, dist, grad_update = self.compute_gradient(attack_instance,
                                                                        neg_instances)
        obj_func_value = attacker_score + self.lambda_val * dist
        obj_function_value_list.append(obj_func_value)

        for iter in range(self.max_iter):
            # no d(x,x_prime) is set to limit the boundary of attacks
            # compute the obj_func_value of the last satisfied instance
            # append to the value list
            # iteration_lst.append(iter)
            # objective_lst.append(obj_func_value)

            past_instance = candidate_attack_instances[-1]
            new_instance = self.update_within_boundary(past_instance, root_instance, grad_update)

            # compute the gradient and objective function value of the new instance
            closer_neg_instances, dist, new_grad_update = \
                self.compute_gradient(new_instance, closer_neg_instances)
            new_attacker_score = self.get_score(new_instance)
            obj_func_value = new_attacker_score + self.lambda_val * dist

            # check convergence information
            # we may reach a local min if the function value does not change
            # if obj_func_value == obj_function_value_list[-1]:
            #    print("Local min is reached. Iteration: %d, Obj value %d" %(iter,obj_func_value))
            #    mat_indices = [x for x in range(0, self.num_features) if new_instance[0][x] != 0]
            #    mat_data = [new_instance[0][x] for x in range(0, self.num_features) if new_
            #    instance[0][x] != 0]
            #    return Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))

            # check a small epsilon(difference is a small value after
            # several iterations)
            if self.check_convergence_info(obj_func_value, obj_function_value_list):
                # print("Goes to Convergence here.... Iteration: %d, Obj value %.4f" %
                # (iter,obj_func_value))
                mat_indices = [x for x in range(0, self.num_features) if new_instance[0][x] != 0]
                mat_data = [new_instance[0][x] for x in range(0, self.num_features) if
                            new_instance[0][x] != 0]

                # plt.plot(iteration_lst,objective_lst)
                return Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))

            # does not satisfy convergence requirement
            # store onto the list
            elif obj_func_value < obj_function_value_list[-1]:
                obj_function_value_list.append(obj_func_value)

            if not (new_instance == candidate_attack_instances[-1]).all():
                candidate_attack_instances.append(new_instance)

            attacker_score = new_attacker_score
            grad_update = new_grad_update

        # print("Convergence has not been found..")
        # plt.plot(iteration_lst, objective_lst)
        mat_indices = [x for x in range(0, self.num_features) if
                       candidate_attack_instances[-1][0][x] != 0]
        mat_data = [candidate_attack_instances[-1][0][x] for x in range(0, self.num_features)
                    if candidate_attack_instances[-1][0][x] != 0]

        return Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))

    def check_convergence_info(self, obj_func_value, obj_function_value_list):
        if len(obj_function_value_list) >= 5:
            val = obj_function_value_list[-5] - obj_func_value
            if val <= self.epsilon:
                return True
        return False

    def compute_gradient(self, attack_instance, neg_instances):
        """
        compute gradient with the trade off of density estimation, return a unit vector
        :param attack_instance:
        :param neg_instances:
        :param lambda_value:
        :param mimicry_params:
        :return: 1.the set of closest legitimate samples 2.the distance value wrt to the closest
                   sample
                 3.the updated gradient by lambda_val * KDE
        """
        grad = self.gradient(attack_instance)
        if self.lambda_val > 0:
            closer_neg_instances, grad_mimicry, dist = \
                self.gradient_mimicry(attack_instance, neg_instances, self.mimicry_params)
            grad_update = grad + self.lambda_val * grad_mimicry
            # print numpy.linalg.norm(grad), numpy.linalg.norm(lambda_value*grad_mimicry)
        else:
            print("The trade_off parameter is 0!")
            closer_neg_instances = neg_instances
            grad_update = grad
            dist = 0

        if (np.linalg.norm(grad_update) != 0):
            grad_update = grad_update / np.linalg.norm(grad_update)
        return closer_neg_instances, dist, grad_update

    def update_within_boundary(self, attack_instance, root_instance, grad_update):
        # find a new instance from the gradient descent step
        # instance = instance - step_size * gradient
        new_instance = np.array(attack_instance - (self.step_size * grad_update))
        for i in range(len(new_instance[0])):
            if (new_instance[0][i] - root_instance[0][i]) > self.bound:

                # print("feature {} in next_pattern: {}".format(i,new_instance[0][i]))
                # print("feature {} in root_attack_instance: {}".format(i,root_instance[0][i]))

                new_instance[0][i] = root_instance[0][i] + self.bound
            elif (new_instance[0][i] - root_instance[0][i]) < - self.bound:

                # print("feature {} in next_pattern: {}".format(i,new_instance[0][i]))
                # print("feature {} in root_attack_instance: {}".format(i,root_instance[0][i]))

                new_instance[0][i] = root_instance[0][i] - self.bound
        return new_instance

    def gradient(self, attack_instance):
        """
        Compute gradient in the case of different classifiers
        if kernel is rbf, the gradient is updated as exp{-2rexp||x-xi||^2}
        if kernel is linear, it should be the weight vector
        support sklearn.svc rbr/linear and robust learner classes
        :return:
        """
        if self.learn_model == SimpleLearner and self.learn_model.model == SVC \
                and self.learn_model.model.kernel == 'rbf':
            grad = []
            dual_coef = self.learn_model.dual_coef_
            support = self.learn_model.support_vectors_
            gamma = self.learn_model.get_params()['gamma']
            kernel = pairwise.rbf_kernel(support, attack_instance, gamma)
            for element in range(0, len(support)):
                if grad == []:
                    grad = (dual_coef[0][element] * kernel[0][element] * 2 * gamma * (
                            support[element] -
                            attack_instance))
                else:
                    grad = grad + (
                            dual_coef[0][element] * kernel[element][0] * 2 * gamma * (
                            support[element] -
                            attack_instance))
            return -grad
        if self.learn_model == SimpleLearner and self.learn_model.model == SVC \
                and self.learn_model.model.kernel == 'linear':
            return self.learn_model.coef_[0]
        else:
            try:
                grad = self.learn_model.get_weight()
                return grad
            except:
                print("Did not find the gradient for this classifier.")

    def gradient_mimicry(self, attack_instance, negative_instances, params):
        # returns:
        # (1) the set of closest legitimate samples;
        # (2) the gradient and
        # (3) the distance value wrt to the closest sample.
        # set default value for max_neg_instance and weight
        max_neg_instance = 10
        weight = 1
        gamma = 0.1
        if 'max_neg_instance' in params.keys():
            max_neg_instance = params['max_neg_instance']
        if 'weight' in params.keys():
            weight = params['weight']
        if 'gamma' in params.keys():
            gamma = params['gamma']
            # select the minicry calculation methods
        if self.mimicry == 'euclidean':
            return self.gradient_euclidean(attack_instance, negative_instances, max_neg_instance,
                                           weight)
        elif self.mimicry == 'kde_euclidean':
            return self.gradient_kde_euclidean(attack_instance, negative_instances,
                                               max_neg_instance, gamma, weight)
        elif self.mimicry == 'kde_hamming':
            return self.gradient_kde_hamming(attack_instance, negative_instances, max_neg_instance,
                                             gamma, weight)
        else:
            print("Gradient Descent Attack: unsupported mimicry distance %s." % self.mimicry)
            return

    def gradient_euclidean(self, attack_instance, negative_instances, max_neg_instance=10,
                           weights=1):
        # compute the euclidean distance of the attack_instance to the negative instances.
        dist = [(negative_instance, self.euclidean_dist(attack_instance, negative_instance
                                                        , weights)) for negative_instance in
                negative_instances]

        # acquire the first max_neg_instance # of best fitted instances
        # sort the resulting dist list according to cmp(a,b).
        dist.sort(key=operator.itemgetter(1))
        if max_neg_instance < len(dist):
            dist = dist[:max_neg_instance]
        neg_instances = [instances[0] for instances in dist]

        # returns:
        # (1) the set of closest legitimate samples;
        # (2) the gradient and
        # (3) the distance value wrt to the closest sample.
        return neg_instances, 2 * (attack_instance - neg_instances[0]), dist[0][1]

    def gradient_kde_euclidean(self, attack_instance, negative_instances, max_neg_instance=10,
                               gamma=0.1, weights=1):
        kernel = [(negative_instance,
                   np.exp(-gamma * self.euclidean_dist_power2(attack_instance, negative_instance,
                                                              weights))) for negative_instance in
                  negative_instances]
        kernel.sort(key=operator.itemgetter(1), reverse=True)
        if max_neg_instance < len(kernel):
            kernel = kernel[:max_neg_instance]
        neg_instances = [instances[0] for instances in kernel]
        kde = 0.0
        gradient_kde = 0.0
        for i, k in enumerate(kernel):
            kde += k[1]
            gradient_kde += -gamma * 2 * (attack_instance - neg_instances[i]) * k[1]

        kde = kde / len(kernel)
        gradient_kde = gamma * gradient_kde / len(kernel)
        # kde estimates similarity, not distance, so -kde is returned
        return neg_instances, -gradient_kde, -kde

    def gradient_kde_hamming(self, attack_instance, negative_instances, max_neg_instance=10,
                             gamma=0.1, weights=1):
        kernel = [(negative_instance,
                   np.exp(-gamma * self.hamming_dist(attack_instance, negative_instance,
                                                     weights))) for negative_instance in
                  negative_instances]

        kernel.sort(key=operator.itemgetter(1), reverse=True)
        if max_neg_instance < len(kernel):
            kernel = kernel[:max_neg_instance]
        neg_instances = [instances[0] for instances in kernel]
        kde = 0.0
        gradient_kde = 0.0
        for i, k in enumerate(kernel):
            kde += k[1]
            # gradient_kde += -gamma*(attack-leg_patterns[i])*k[1]
            gradient_kde += -(attack_instance - neg_instances[i]) * k[1]

        kde = kde / len(kernel)
        # gradient_kde = gamma*gradient_kde / len(kernel)
        # I'm using minus since kde estimates similarity, not distance
        return neg_instances, -gradient_kde, -kde

    # basic distance caclulation functions
    def hamming_dist(self, a, b, norm_weights):
        return np.sum(np.absolute(a - b) * norm_weights)

    def euclidean_dist_power2(self, a, b, norm_weights):
        return np.sum(((a - b) * norm_weights) ** 2)

    def euclidean_dist(self, a, b, norm_weights):
        return np.linalg.norm((a - b) * norm_weights)

    def get_score(self, pattern):
        score = self.learn_model.decision_function(pattern)
        return score[0]
