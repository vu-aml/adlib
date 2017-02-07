from adversaries.adversary import Adversary
from typing import List, Dict
from data_reader.input import Instance, FeatureVector
import numpy as np
from learners.learner import InitialPredictor
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

class CostSensitive(Adversary):
    def __init__(self, learner=None):
        self.Ua = Ua or None
        self.Vi = Vi or None
        self.Uc = Uc or None
        self.Wi = Wi or None
        self.Xc = None
        self.Xdomain = None
        self.positive_instances = None
        self.delta_Ua = None
        self.num_features = None
        self.learner = learner    #type: Classifier

    def attack(self, instances) -> List[Instance]:
        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == InitialPredictor.positive_classification:
                transformed_instances.append(self.a(transformed_instance))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def set_params(self, params: Dict):
        if params['measuring_cost'] is not None:
            self.Vi = params['measuring_cost']

        if params['adversary_utility'] is not None:
            self.Ua = params['adversary_utility']

        if params['transform_cost'] is not None:
            self.Wi = params['transform_cost']

        if params['scenario'] is not None:
            self.scenario = params['scenario']

    def get_available_params(self) -> Dict:
        params = {'measuring_cost': self.Vi,
                  'adversary_utility': self.Ua,
                  'transform_cost': self.Wi,
                  'scenario': 'All_Word'}
        return params

    def set_adversarial_params(self, learner, train_instances):

        self.learner = learner
        self.Xc = train_instances
        self.num_features = train_instances[0].get_feature_vector.get_feature_count()
        self.positive_instances = [x for x in train_instances if x.get_label() == InitialPredictor.positive_classification]
        self.delta_Ua = self.Ua[0][1] - self.Ua[1][1]

    def find_mcc(self,i, w):
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
        if i==0:
            return float('inf'), None
        minCost = float('inf')
        minList = []
        # need to figure out what I'm calling the domain of a given feature
        for xi_prime in self.Xdomain[i]:
            delta_log_odds = log_odds(i, xi_prime) - log_odds(i, self.Xc[i])
            if delta_log_odds >= 0:
                curCost, curList = self.find_mcc(i-1, w - delta_log_odds)
                curCost += self.Wi(self.Xc[i], xi_prime)
                curList += [(i,xi_prime)]
                if curCost < minCost:
                    minCost = curCost
                    minList = curList
        return minCost,minList

    # should I be passing an index here as well or just the feature vector...
    def log_odds(self, x):
        '''
        Args: x (Instance)
        return: log P(+|x) / P(-|x)
        '''
        try:
            log_prob = self.learner.predict_log_proba(x)
        except:
            print("This adversary currently only supports probability models")
            raise

        return log_prob[0,1]/log_prob[0,0]

    def gap(self, x):
        '''
        The gap is defined as the difference between the log odds of the instance
        and the log threshold that needs to be reached for the classifier to
        classify the positive instance as negative

        Args: x (Instance)
        return: LOc(x) - LT(Uc)
        '''
        return log(self.log_odds(x) - self.log_threshold(self.Uc))

    def log_threshold(self, Uc):
        return (Uc[0][0] - Uc[1][0]) / (Uc[1][1] - Uc[0][1])

    def a(x):
        '''
        Change instance x only if the minimum cost to effectively fool C is
        less than delta_Ua: the user defined utility gained by fooling
        classifier

        Args: x (Instance)
        return: possible
        '''
        W = self.gap(x) # discretized
        minCost,minList = find_mcc(len(x),W)
        if minCost < self.delta_Ua:
            for i,xi_prime in minList:
                x.get_feature_vector().swap_feature(i,xi_prime)
        return x
