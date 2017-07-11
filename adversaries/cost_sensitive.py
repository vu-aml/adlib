from adversaries.adversary import Adversary
from typing import List, Dict
from data_reader.binary_input import Instance, FeatureVector
import numpy as np
from learners.learner import learner
from math import log
from copy import deepcopy

'''Adversarial Classification based on paper by:
    Dalvi, Domingos, Mausam, Sanghai, and Verma
    University of Washington

Concept:
    Given complete information about initial classifier C and an instance X, adversary
    finds a feature change strategy A(x) that maximizes its own utility Ua.
    By modeling the problem as a COP, which can be formulated as an integer
    linear program, the adversary finds the minimum cost camoflauge (MCC),
    or smallest cost of feature changes that covers the log odds gap between
    classifier classifying the instance as positive and classifying it as a
    false negative.  Only changes instances the classifier classified as
    positive and that can be changed without costing so much that it
    outweighs the utility that would be gained by C falsely classifying the instance as
    negative.  Runs in pseudo linear time.

    TODO: Extend compatibility beyond probability classifier models
'''

#the parameters can be set according to the experiments described in the paper
# position: (-,-)= (0,0) (-,+) = (0,1) (+,-)= (1,0) (+,+)= (1,1)
class CostSensitive(Adversary):
    def __init__(self, Ua = None, Vi = None, Uc = None, Wi = None, learner=None):
        self.Ua = Ua
        self.Vi = Vi
        self.Uc = Uc
        self.Xc = None
        self.Xdomain = [0,1]   #all the features are binary, so possible values are either 0 or 1
        self.positive_instances = None
        self.delta_Ua = None
        self.num_features = None
        self.learn_model = learner    #type: Classifier
        self.scenario = "All_Word"

    def attack(self, instances) -> List[Instance]:
        transformed_instances = []
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == learner.positive_classification:
                transformed_instances.append(self.a(transformed_instance))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def set_params(self, params: Dict):
        if params['Vi'] is not None:
            self.Vi = params['Vi']

        if params['Ua'] is not None:
            self.Ua = params['Ua']

        if params['scenario'] is not None:
            self.scenario = params['scenario']

    def set_classifier_utility(self,Uc):
        self.Uc = Uc

    def get_available_params(self) -> Dict:
        params = {'measuring_cost': self.Vi,
                  'adversary_utility': self.Ua,
                  'transform_cost': self.Wi,
                  'classifier_utility':self.Uc,
                  'scenario': 'All_Word'}
        return params

    def set_adversarial_params(self, learner, train_instances):

        self.learn_model = learner
        self.Xc = train_instances
        self.num_features = train_instances[0].get_feature_vector().get_feature_count()
        self.positive_instances = [x for x in train_instances if x.get_label() == learner.positive_classification]
        self.delta_Ua = self.Ua[0][1] - self.Ua[1][1]

    def find_mcc(self,i, w, x: Instance):
        '''
        Given number of features to be considered and the gap to be filled such
        that classifier will classify given positive instance as negative, recursively
        compute the minimum cost and changes to be made to transform the instance
        feature by feature such that the gap is filled.

        Args: i (int): number of features to be considered
              w (int): gap to be filled
        return: 1) minimum cost to transform the instance
                2) a list of pairs of original feature indices and their
                   corresponding transformations
        '''
        if w<=0:
            return 0,[]
        if i<0:
            return 0,[]
        minCost = float('inf')
        minList = []
        # need to figure out what I'm calling the domain of a given feature
        for xi_prime in self.Xdomain:
            instance_prime = deepcopy(x)
            instance_prime.get_feature_vector().change_bit(i,xi_prime)
            delta_log_odds = self.log_odds(instance_prime) - self.log_odds(x)
            if delta_log_odds >= 0:
                curCost, curList = self.find_mcc(i-1, w - delta_log_odds,x)
                curCost += self.w(x,instance_prime,i)
                curList += [(i,xi_prime)]
                if curCost < minCost:
                    minCost = curCost
                    minList = curList
        return minCost , minList

    def gap(self,x):
        '''
        The gap is defined as the difference between the log odds of the instance
        and the log threshold that needs to be reached for the classifier to
        classify the positive instance as negative

        Args: x (Instance)
        return: LOc(x) - LT(Uc)

        '''
        return self.log_odds(x) - self.log_threshold()


    def log_odds(self, x):
        '''
        Args: x (Instance)
        return: log P(+|x) / P(-|x)
        '''
        try:
            log_prob = self.learn_model.predict_log_proba(x)
        except:
            print("This adversary currently only supports probability models")
            raise
        else:
            return log_prob[0,1] - log_prob[0,0]



    def log_threshold(self, Uc = None):
        if Uc == None:
            return (self.Uc[0][0] - self.Uc[1][0]) / (self.Uc[1][1] - self.Uc[0][1])
        else:
            return (Uc[0][0] - Uc[1][0]) / (Uc[1][1] - Uc[0][1])


    def a(self,x):
        '''
        Change instance x only if the minimum cost to effectively fool C is
        less than delta_Ua: the user defined utility gained by fooling
        classifier

        Args: x (Instance)
        return: possible
        '''
        W = self.gap(x) # discretized
        minCost,minList = self.find_mcc(self.num_features,W, x)
        if minCost < self.delta_Ua:
            for i,xi_prime in minList:
                x.get_feature_vector().change_bit(i,xi_prime)
        return x


    def w(self,x, x_prime,i):
       w = x.get_feature_vector().get_feature(i) - x_prime.get_feature_vector().get_feature(i)
       return w
