# dp_learner_test.py
# Tests data poisoning learners
# Matthew Sedam. 2018.

from data_reader.binary_input import Instance
from typing import Dict, List


class TestDataPoisoningLearner:

    def __init__(self, learner_name: str,
                 training_instances: List[Instance],
                 testing_instances: List[Instance],
                 attack_instances: List[Instance],
                 params: Dict = None,
                 verbose=True):
        """
        Test setup.
        :param learner_name: Either 'trim', 'atrim', or 'irl'
        :param training_instances: the training instances
        :param testing_instances: the testing instances
        :param attack_instances: the attacked instances
        :param params: the params to pass to the learner - if None, defaults
                       will be used
        :param verbose: if True, will print START and STOP and set learners to
                        verbose mode
        """

        if learner_name.lower() not in ['trim', 'atrim', 'irl']:
            raise ValueError('Learner name not trim, atrim, nor irl.')

        self.learner_name = learner_name
        self.training_instances = training_instances
        self.testing_instances = testing_instances
        self.attack_instances = attack_instances
        self.params = params
        self.verbose = verbose

    def test(self):
        pass
