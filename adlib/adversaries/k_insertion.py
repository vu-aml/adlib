# k_insertion.py
# An implementation of the k-insertion attack where the attacker adds k data
# points to the model
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
import cvxpy as cvx
import numpy as np
from copy import deepcopy
from progress.bar import Bar
from typing import List, Dict


# TODO: Implement gradient functions
# TODO: Implement gradient descent for 1 added vector
# TODO: Implement loop for k vectors using gradient descent


class KInsertion(Adversary):
    def __init__(self):
        Adversary.__init__(self)

    def attack(self, instances) -> List[Instance]:
        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
