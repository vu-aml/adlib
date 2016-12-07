from adversaries.adversary import AdversaryStrategy
from typing import List, Dict
from data_reader.input import Instance, FeatureVector
import numpy as np
from learners.learner import InitialPredictor
import math

class Adversary(AdversaryStrategy):
    def __init__(self):
        self.Ua = None
        self.optim_strategy = None
        self.Vi = None
        self.Uc = None
        self.Wi = None
        self.learn_model = None
        self.Xc = None
        self.Xdomain = None
        self.positive_instances = None
        self.delta_Ua = None

    def change_instances(self, instances) -> List[Instance]:
        for instance in instances:
			transformed_instance = deepcopy(instance)
			if instance.get_label() == InitialPredictor.positive_classification:
				transformed_instances.append(self.A(transformed_instance))
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
        self.learn_model = learner
        self.Xc = train_instances
        self.positive_instances = [x for x in train_instances if x.get_label() == InitialPredictor.positive_classification]
        self.delta_Ua = self.Ua[0][1] - self.Ua[1][1]
        self.Xdomain = [0,1]

    def find_mcc(self,i, w):
        '''
        input: 1) number of features to be considered and 2) gap to be filled
               such that classifier will classify positive instance as negative
        return: 1) minimum cost to transform the instance and 2) a list of pairs
                of original feature indices and their corresponding transformations
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
                # maybe implement curList as a dictionary instead
                curList += [(i,xi_prime)]
                if curCost < minCost:
                    minCost = curCost
                    minList = curList
        return minCost,minList

    # should I be passing an index here as well or just the feature vector...
    def log_odds(self, x):
        '''
        input: feature vector x
        return: log P(+|x) / P(-|x)
        '''
        log_prob = self.learn_model.predict_log_proba(x)
        return log_prob[0,1]/log_prob[0,0]

    def gap(self, x):
        '''
        The gap is defined as the difference between the log odds of the instance
        and the log threshold that needs to be reached for the classifier to
        classify the positive instance as negative
        input: instance x
        return: LOc(x) - LT(Uc)
        '''
        return math.log(self.log_odds(x) - self.log_threshold(self.Uc))

    def log_threshold(self, Uc = self.Uc):
        return (Uc[0][0] - Uc[1][0]) / (Uc[1][1] - Uc[0][1])


    def A(x):
        W = self.gap(x) # discretized
        minCost,minList = find_mcc(len(x),W)
        if minCost < self.delta_Ua:
            new_x = x
            for i,xi_prime in minList:
                x[i] = xi_prime
        return x
