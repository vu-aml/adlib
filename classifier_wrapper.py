from data_reader import input
from data_reader.input import Instance
from adversaries.adversary import AdversaryStrategy
from learners.learner import InitialPredictor, ImprovedPredictor
from learners.models.model import BaseModel
from learners.models import sklearner
import factories
from typing import Dict, List


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
    def __init__(self, model_alg, data_name=None, defence_strategy=None, test_fraction=None):
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
        self.num_features = self.train_instances[0].feature_vector.feature_count
        self.predictions = None
        self.defence_strategy = defence_strategy
        self.model = set_model(model_alg)
        self.simulated_learner = None
        self.simulated_adversary = None
        self.simulated_defence = None
        self.test_fraction = None
        if self.defence_strategy is not None:
            self.simulated_learner = set_simulated_learner(defence_strategy[0])
            self.simulated_learner.set_model(self.model)
            self.simulated_adversary = set_adversary(defence_strategy[1])
            self.simulated_defence = set_learner_improve(defence_strategy[0])

    def set_train_data(self, data_name):
        """
        set training data instances
        Args:
            data_name: file name of training data
        """
        self.train_instances = set_train_instances(data_name)

    def set_test_data(self, data_name):
        """
        set testing data instances
        Args:
            data_name: file name of testing data
        """
        self.test_instances = set_test_instances(data_name)

    def set_num_features(self, num_features):
        """

        Args:
            num_features: new count of features for a feature vecto

        """
        self.num_features = num_features

    def train(self):
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

    def predict(self, instances=None):
        """
        returns prediction labels for specified instances,
        if no instances specified return prediction for self.test_instances
        Args:
            instances: optional List[instance] to be predicted

        Returns: List of class label as predictions

        """
        if instances is None:
            if self.defence_strategy is None:
                return self.model.predict(self.test_instances)
            else:
                return self.simulated_defence.predict(self.test_instances)
        else:
            if self.defence_strategy is None:
                return self.model.predict(instances)
            else:
                return self.simulated_defence.predict(instances)

    def predict_proba(self, instances=None):
        """
        returns prediction probability values for specified instances,
        (i.e. the probability for a instance to be benign)
        if no instances specified return prediction for self.test_instances
        Args:
            instances: optional List[instance] to be predicted

        Returns: List of probability values as predictions

        """
        if instances is None:
            if self.defence_strategy is None:
                return self.model.predict_proba_adversary(self.test_instances)
            else:
                return self.simulated_defence.predict_proba(self.test_instances)
        else:
            if self.defence_strategy is None:
                return self.model.predict_proba_adversary(instances)
            else:
                return self.simulated_defence.predict_proba(instances)

    def decision_function(self, instances: List[Instance]):
        if self.defence_strategy is None:
            return self.model.decision_function_adversary(instances)
        else:
            return self.simulated_defence.decision_function(instances)


    """
    a series of setter methods that configs model and attack simulation
    """

    def set_model_params(self, params: Dict):
        self.model.set_params(params)

    def set_simulated_learner_params(self, params: Dict):
        self.simulated_learner.set_params(params)

    def set_simulated_adversary_params(self, params: Dict):
        self.simulated_adversary.set_params(params)

    def set_defence_params(self, params: Dict):
        self.simulated_defence.set_params(params)
