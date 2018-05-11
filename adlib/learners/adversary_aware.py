from typing import Dict
from sklearn.naive_bayes import GaussianNB
from adversaries.cost_sensitive import CostSensitive
from data_reader.operations import fv_equals
from adlib.learners.simple_learner import SimpleLearner
from sklearn.utils.validation import check_array,check_is_fitted
import numpy as np
from copy import deepcopy

"""
   Based on the Adversarial Classification By Nilesh Dalvi, 
   Pedro Domingos, Mausam, Sumit Sanghai, Deepak Verma. 
   This learner is set based on the assumption that the attacker
   is a naive bayes based attacker and uses it s optimal strategy to modify test 
   instances. 
"""

class AdversaryAware(object):
    def __init__(self, training_instances=None, attacker=None):
        """
        :param training_instances:
        :param attacker: need to be CostSensitive Attacker. Otherwise,
                         we need to import utility of attacker or utility of learner.
        """
        learner_model = GaussianNB()
        self.learner = SimpleLearner(model= learner_model,training_instances=training_instances)
        self.attacker = attacker  # Type: Adversary
        self.training_instances = training_instances
        if type(attacker) == CostSensitive:
            self.Uc = attacker.Uc
            self.Ua = attacker.Ua

    def set_params(self, params: Dict):
        if 'attacker' in params.keys():
            self.attacker = params['attacker']
            if type(params['attacker']) == CostSensitive:
                 self.Uc = self.attacker.Uc
                 self.Ua = self.attacker.Ua
        if 'Ua' in params.keys():
            self.Ua = params['Ua']
        if 'Uc' in params.keys():
            self.Uc = params['Uc']

    def get_params(self):
        return {'attacker': self.attacker,
                'Uc': self.Uc,
                'Ua': self.Ua}

    def train(self):
        """
        train the untampered NaiveBayes classifier.
        :param: list of instances, training set
        :return: None
        """
        self.learner.train()

    def predict(self, instances):
        result = []
        attacker_instance = self.attacker.attack(self.learner.training_instances)
        self.find_new_list(attacker_instance)
        for instance in instances:
            result.append(self.compute_c_x(instance))
        return result


    def compute_c_x(self,instance):
        """
        algorithm 3 in cost sensitive attack
        using NB classifier and the camouflage to compute the uility
        :param instance: value needed to be classified
        :return: 1 if malicious, -1 if benign
        """
        #find P(-) and P(+)
        #Px- = P(-)*P(x|-)
        #Px+ = P(+)*Pa(x|+)
        class_prior = self.learner.model.learner.class_prior_
        p_x_negative = class_prior[0] * np.exp(self.posterior_proba(instance))[0,0]
        p_x_positive = class_prior[1] * self.compute_p_a_prime(instance)
        u_positive_x = p_x_positive * self.Uc[1][1] + p_x_negative * self.Uc[0][1]
        u_negative_x= p_x_positive * self.Uc[1][0] + p_x_negative * self.Uc[0][0]
        if u_positive_x > u_negative_x:
            return 1
        else:
            return -1

    def compute_p_a_prime(self,instance):
        #find X'A(x')
        new_list = []
        for attack_instance in self.new_list:
            if not fv_equals(attack_instance.get_feature_vector(),instance.get_feature_vector()):
                new_list.append(attack_instance)
        p_x_ = 0
        for attack_instance in new_list:
            p_x_ += np.exp(self.posterior_proba(attack_instance))[0,1]
        return p_x_+ self.i_x(instance) * np.exp(self.posterior_proba(instance))[0,1]


    def i_x(self,instance):
        if self.learner.predict(instance) == -1:
            return 1
        if self.attacker.w_feature_difference(instance,self.attacker.a(instance)) \
                >= (self.Ua[0][1] - self.Ua[1][1]):
            return 1
        return 0

    def find_new_list(self, instances):
        """
        Find the changed instances in the attacker's modified data from training data
        :param instances: attacker's modified data
        :return: None
        """
        new_list = []
        for attack_instance in instances:
            equal = False
            for instance in self.training_instances:
                if fv_equals(attack_instance.get_feature_vector(),instance.get_feature_vector()):
                    equal = True
                    break
            if not equal:
                new_list.append(attack_instance)
        self.new_list = new_list

    def posterior_proba(self,x):
        """
        Find the posterior proba given an instance x.
        :param x:
        :return:
        """
        X = x.get_feature_vector().get_csr_matrix().toarray()
        check_array(X)
        check_is_fitted(self.learner.model.learner, "classes_")
        joint_likelihood = []
        for i in range(np.size(self.learner.model.learner.classes_)):
            n_ij = np.sum(np.log(1 / (np.sqrt(2. * np.pi * self.learner.model.learner.sigma_[i, :] ** 2))))
            n_ij -= 0.5 * np.sum(((X - self.learner.model.learner.theta_[i, :]) ** 2) /
                                 (self.learner.model.learner.sigma_[i, :]), 1)
            joint_likelihood.append(n_ij)
        joint_likelihood = np.array(joint_likelihood).T
        return joint_likelihood


