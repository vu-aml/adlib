# data_modification.py
# An implementation of the data modification attack where the attacker modifies
# certain feature vectors
# Matthew Sedam

from adlib.adversaries.adversary import Adversary
from data_reader.binary_input import Instance
from copy import deepcopy
from typing import List, Dict


class DataModification(Adversary):
    def __init__(self, learner, verbose=False):
        Adversary.__init__(self)
        self.learner = deepcopy(learner)
        self.verbose = verbose

    def attack(self, instances) -> List[Instance]:
        if len(instances) == 0:
            raise ValueError('Need at least one instance.')

        raise NotImplementedError

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
