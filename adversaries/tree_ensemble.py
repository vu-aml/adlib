import pickle
from data_reader.input import Instance
from typing import List, Dict


class Adversary(object):

	def attack(self, instances) -> List[Instance]:
		raise NotImplementedError

	def set_params(self, params: Dict):
		raise NotImplementedError

	def get_available_params(self) -> Dict:
		raise NotImplementedError

	def set_adversarial_params(self, learner, train_instances):
		raise NotImplementedError
