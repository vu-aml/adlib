from adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from data_reader.real_input import RealFeatureVector
from learners.simple_learner import SimpleLearner
from typing import List, Dict
from random import shuffle
import numpy as np
import learners as learners
from copy import deepcopy
from sklearn.svm import SVC
from sklearn.metrics import pairwise



""" A simpler gradient descent based evasion attack.
    Solve the problem: min g(x) s.t. |xi-x| <= dmax.
"""



class SimpleGradientDescent(Adversary):
    def __init__(self, learn_model=None, step_size=0.01, trade_off=10,
                 stp_constant=0.0000001,max_iter=1000, bound = 0.1, binary = False, all_malicious = False):
        """
        :param learner: Learner(from learners)
        :param max_change: max times allowed to change the feature
        :param epsilon: the limit of difference between transform costs of ,xij+1, xij, and orginal x
        :param step_size: weight for coordinate descent
        :param max_iter: maximum number of gradient descent iterations to be performed
                               set it to a large number by default.
        :param bound: limit how far one instance can move from its root instance.
                      this is d(x,x_prime) in the algortihm.
        """
        Adversary.__init__(self)
        self.step_size = step_size
        self.num_features = 0
        self.learn_model = learn_model
        self.epsilon = stp_constant
        self.max_iter = max_iter
        self.bound = bound
        self.binary = binary
        self.all_malicious = all_malicious

    def get_available_params(self) -> Dict:
        return {'step_size': self.step_size,
                'trade_off': self.lambda_val,
                'learn_model': self.learn_model,
                'stp_constant': self.epsilon,
                'max_iter': self.max_iter,
                'bound' : self.bound,
                'binary':self.binary}

    def set_params(self, params: Dict):
        if 'step_size' in params.keys():
            self.step_size = params['step_size']
        if 'trade_off' in params.keys():
            self.lambda_val = params['trade_off']
        if 'stp_constant' in params.keys():
            self.epsilon = params['stp_constant']
        if 'learn_model' in params.keys():
            self.learn_model = params['learn_model']
        if 'max_iter' in params.keys():
            self.max_iter = params['max_iter']
        if  'bound' in params.keys():
            self.bound = params['bound']
        if 'binary' in params.keys():
            self.binary = params['binary']

    def set_adversarial_params(self, learner, train_instances: List[Instance]):
        self.learn_model = learner
        self.num_features = train_instances[0].get_feature_count()
        self.train_instances = train_instances

    def attack(self, Instances) -> List[Instance]:
        if self.num_features == 0:
            self.num_features = Instances[0].get_feature_count()
        transformed_instances = []
        for instance in Instances:
            if instance.label < 0:
                transformed_instances.append(instance)
            else:
                if self.binary:
                    transformed_instances.append(self.binary_gradient_descent(instance))
                else:
                    transformed_instances.append(self.gradient_descent(instance))
        #plt.show()
        return transformed_instances


    def binary_gradient_descent(self, attack_instance:Instance):
        #sparse attack with binary features
        index_lst = []
        iter_time = 0
        attacker_score = self.get_score(attack_instance.get_csr_matrix().toarray())
        while iter_time < self.max_iter:
            grad = self.gradient(attack_instance.get_csr_matrix().toarray())
            if index_lst is not []:
               #eliminate the index we have already modified
               for i in index_lst:
                   grad[i] = 0
            change_index = np.argmax(np.absolute(grad))
            new_attack_instance = deepcopy(attack_instance)
            new_attack_instance.get_feature_vector().flip_bit(change_index)
            index_lst.append(change_index)

            new_attacker_score = self.get_score(new_attack_instance.get_csr_matrix().toarray())
            if new_attacker_score < attacker_score:
                attack_instance = new_attack_instance
                attacker_score = new_attacker_score
                iter_time += 1
        return attack_instance


    def gradient_descent(self, instance: Instance):
        #store iteration and objective values for plotting....
        #iteration_lst = []
        #objective_lst = []

        # attack_intance-> np array
        attack_instance = instance.get_csr_matrix().toarray()
        root_instance = attack_instance
        obj_function_value_list = []

        # store the modified gradient descent attack instances
        # find a list of potential neg_instances, the closest distance, and updated gradients
        candidate_attack_instances = [attack_instance]
        attacker_score = self.get_score(attack_instance)
        grad = self.gradient(attack_instance)
        obj_function_value_list.append(attacker_score)

        for iter in range(self.max_iter):
            # no d(x,x_prime) is set to limit the boundary of attacks
            # compute the obj_func_value of the last satisfied instance
            # append to the value list
            #iteration_lst.append(iter)
            #objective_lst.append(attacker_score)

            past_instance = candidate_attack_instances[-1]
            new_instance = self.update_within_boundary(past_instance,root_instance,grad)
            grad = self.gradient(new_instance)
            new_attacker_score = self.get_score(new_instance)
            # check convergence information
            # we may reach a local min if the function value does not change
            # if obj_func_value == obj_function_value_list[-1]:
            #    print("Local min is reached. Iteration: %d, Obj value %d" %(iter,obj_func_value))
            #    mat_indices = [x for x in range(0, self.num_features) if new_instance[0][x] != 0]
            #    mat_data = [new_instance[0][x] for x in range(0, self.num_features) if new_instance[0][x] != 0]
            #    return Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))

            # check a small epsilon(difference is a small value after
            # several iterations)
            if self.check_convergence_info(new_attacker_score, obj_function_value_list):
                print("Goes to Convergence here.... Iteration: %d, Obj value %.4f" % (iter,attacker_score))
                mat_indices = [x for x in range(0, self.num_features) if new_instance[0][x] != 0]
                mat_data = [new_instance[0][x] for x in range(0, self.num_features) if new_instance[0][x] != 0]

                #plt.plot(iteration_lst,objective_lst)
                return Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))

            # does not satisfy convergence requirement
            # store onto the list
            elif new_attacker_score < obj_function_value_list[-1]:
                obj_function_value_list.append(new_attacker_score)

            if not (new_instance == candidate_attack_instances[-1]).all():
                candidate_attack_instances.append(new_instance)

        print("Convergence has not been found..")
        #plt.plot(iteration_lst, objective_lst)
        mat_indices = [x for x in range(0, self.num_features) if candidate_attack_instances[-1][0][x] != 0]
        mat_data = [candidate_attack_instances[-1][0][x] for x in range(0, self.num_features)
                    if candidate_attack_instances[-1][0][x] != 0]


        return Instance(-1, RealFeatureVector(self.num_features, mat_indices, mat_data))

    def check_convergence_info(self, obj_func_value, obj_function_value_list):
        if len(obj_function_value_list) >= 5:
            val = obj_function_value_list[-5] - obj_func_value
            if val <= self.epsilon:
                return True
        return False



    def update_within_boundary(self,attack_instance, root_instance, grad_update):
        # find a new instance from the gradient descent step
        # instance = instance - step_size * gradient
        new_instance = np.array(attack_instance - (self.step_size * grad_update))
        for i in range(len(new_instance[0])):
            if (new_instance[0][i] - root_instance[0][i]) > self.bound:

                #print("feature {} in next_pattern: {}".format(i,new_instance[0][i]))
                #print("feature {} in root_attack_instance: {}".format(i,root_instance[0][i]))

                new_instance[0][i] = root_instance[0][i] + self.bound
            elif (new_instance[0][i] - root_instance[0][i]) < - self.bound:

                #print("feature {} in next_pattern: {}".format(i,new_instance[0][i]))
                #print("feature {} in root_attack_instance: {}".format(i,root_instance[0][i]))

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
        if type(self.learn_model) == SimpleLearner and type(self.learn_model.model.learner) == SVC:
            param_map = self.learn_model.get_params()
            attribute_map = self.learn_model.get_attributes()
            if param_map["kernel"] == "rbf":
                grad = []
                dual_coef = attribute_map["dual_coef_"]
                support = attribute_map["support_vectors_"]
                gamma = param_map["gamma"]
                kernel = pairwise.rbf_kernel(support, attack_instance, gamma)
                for element in range(0, len(support)):
                    if grad == []:
                        grad = (dual_coef[0][element] * kernel[0][element] * 2 * gamma * (support[element] -
                                                                                          attack_instance))
                    else:
                        grad = grad + (
                            dual_coef[0][element] * kernel[element][0] * 2 * gamma * (support[element] -
                                                                                      attack_instance))
                return np.array(-grad)
            if param_map["kernel"] == "linear":
                return np.array(attribute_map["coef_"][0])
        else:
            try:
                grad = self.learn_model.get_weight()
                return np.array(grad)
            except:
                print("Did not find the gradient for this classifier.")


    def get_score(self, pattern):
        score = self.learn_model.decision_function(pattern)
        return score[0]
