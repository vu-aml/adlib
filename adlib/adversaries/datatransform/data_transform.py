# data_transform.py
# A data-transformation implementation based on IEEE S&P 2018 paper
# 'Manipulating Machine Learning: Poisoning Attacks and Countermeasures for
# Regression Learning.'
# Matthew Sedam. 2018. Original code from Matthew Jagielski.

from adlib.adversaries import Adversary
from data_reader.binary_input import Instance
from typing import Dict, List


class DataTransform(Adversary):
    def __init__(self):
        Adversary.__init__(self)
        raise NotImplementedError

    def attack(self, instances) -> List[Instance]:
        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
