from data_reader import input
from adversaries.adversary import AdversaryStrategy
from learners.learner import InitialPredictor, ImprovedPredictor
from learners.models.model import BaseModel
from learners.models import sklearner
import factories
from typing import Dict


def set_train_instances(data):
    """
    Returns a list of Instance given data file name
    Args:
        data: str of data file name

    Returns: List of Instance

    """
    return input.load_instances([data, 'train'])


def set_test_instances(data):
    """
    Returns a list of Instance given data file name
    Args:
        data: str of data file name

    Returns: List of Instance

    """
    return input.load_instances([data, 'test'])


def set_model(model_alg) -> BaseModel:
    """
    Find and initialize learner model
    Args:
        model_alg: string of model type, or model object from sklearn

    Returns: a model object of specified model

    """
    if not type(model_alg) is str:
        return sklearner.Model(model_alg)
    else:
        factory = factories.ModelFactory(model_alg)
        factory.load_class()
        return factory.get_class()


def set_simulated_learner(learner_model_alg) -> InitialPredictor:
    """
    Find and initialize learner for the simulated attack and defence
    Args:
        learner_model_alg: string of model type

    Returns: a model object of specified model

    """
    factory = factories.InitialPredictorFactory(learner_model_alg)
    factory.load_class()
    return factory.get_class()


def set_adversary(adversary_alg) -> AdversaryStrategy:
    """
    Find and initialize adversary for the simulated attack and defence
    Args:
        adversary_alg:  string of adversary type

    Returns: a object of type AdversaryStrategy

    """
    if adversary_alg is None:
        return None
    else:
        factory = factories.AdversaryFactory(adversary_alg)
        factory.load_class()
        return factory.get_class()


def set_learner_improve(learner_improve_alg) -> ImprovedPredictor:
    """
    Find and initialize improved learner for the simulated attack and defence
    Args:
        learner_improve_alg:  string of adversary type

    Returns: a object of type AdversaryStrategy

    """
    if learner_improve_alg is None:
        return None
    else:
        factory = factories.ImprovedPredictorFactory(learner_improve_alg)
        factory.load_class()
        return factory.get_class()


class Classifier(object):
    """
    Wrapper of both naive and robust classifier.
    Provides basic interface for classification.
    Allows declaration  of internal defence strategy that trains model against evasion attacks
    """
    def __init__(self, model_alg, data_name=None, defence_strategy=None):
        """

        Args:
            model_alg: string of type of model, or model object from sklearn
                        (currently only supports model object from sklearn)
            data_name:  optional file name for instance initialization (string type)
            defence_strategy: optional defence strategy of string type
                        (currently only support nash and stackelberg)
        """
        self.data = data_name
        self.train_instances = set_train_instances(data_name)
        self.test_instances = set_test_instances(data_name)
        self.predictions = None
        self.defence_strategy = defence_strategy
        self.model = set_model(model_alg)
        self.simulated_learner = None
        self.simulated_adversary = None
        self.simulated_defence = None

        if self.defence_strategy is not None:
            self.simulated_learner = set_simulated_learner(defence_strategy)
            self.simulated_learner.set_model(self.model)
            self.simulated_adversary = set_adversary(defence_strategy)
            self.simulated_defence = set_learner_improve(defence_strategy)

    def set_training_data(self, data_name):
        """
        set training data instances
        Args:
            data_name: file name of training data
        """
        self.train_instances = set_train_instances(data_name)

    def train(self,):
        """
        Trains internal model, if a defence_strategy is declared, then improve learner
        """
        if self.defence_strategy is None:
            self.model.train(self.train_instances)
        else:
            self.simulated_learner.train(self.train_instances)
            self.simulated_adversary.set_adversarial_params(self.simulated_learner,
                                                            self.train_instances)
            self.simulated_defence.set_adversarial_params(self.simulated_learner,
                                                          self.simulated_adversary)
            self.simulated_defence.improve(self.train_instances)

    def predict(self):
        """
        predict classification result of test instances
        Returns: list of binary classified result

        """
        if self.defence_strategy is None:
            return self.model.predict(self.test_instances)
        else:
            return self.simulated_defence.predict(self.test_instances)

    def retrain(self):
        # TODO refactor learners.retraining and move it here
        raise NotImplementedError

    """
    a series of setter methods that configs model and attack simulation
    """

    def set_model_params(self, params: Dict):
        self.model.set_params(params)

    def set_simulated_adversary_params(self, params: Dict):
        self.simulated_adversary.set_params(params)

    def set_defence_params(self, params: Dict):
        self.simulated_defence.set_params(params)



