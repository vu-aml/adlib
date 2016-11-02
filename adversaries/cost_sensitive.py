from adversaries.adversary import AdversaryStrategy
from typing import List, Dict
from data_reader.input import Instance, FeatureVector
import numpy as np

class Adversary(AdversaryStrategy):
    def __init__(self):
        self.Ua = None
        self.optim_strategy = None
        self.Vi = None
        self.Uc = None
        self.Wi = None
        self.learner = None
        self.Xc = None
        self.Xdomain = None

    def change_instances(self, instances) -> List[Instance]:

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


    def findMCC(self,i, w):
        if w<=0:
            return (0,[])
        minCost = float('inf')
        minList = []
        for xi_prime in self.Xdomain[i]:
            l_odds = logOdds(i, xi_prime)
            if l_odds >= 0:
                (curCost, curList) = self.findMCC(i-1, w - l_odds)
                curCost += self.Wi(self.Xc[i], xi_prime)
                curList += [(i,xi_prime)]
                if curCost < minCost:
                    minCost = curCost
                    minList = curList
        return (minCost,minList)
    def logOdds(self, x):
