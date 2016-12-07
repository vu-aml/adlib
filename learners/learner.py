from typing import List, Dict
from adversaries.adversary import AdversaryStrategy
from learners.models.model import BaseModel


class InitialPredictor(object):
    """Base class for initial learning methods.

    Defines the bare-minimum functionality for initial learning
    strategies. Specified learning algorithms can create wrappers
    around the underlying methods.

    """
    # Static variables for positive/negative classiciation so we don't have to use hard coded values
    positive_classification = 1
    negative_classification = -1

    def __init__(self):
        """New generic initial learner with no specified learning model.

        """
        self.model = None  # type: BaseModel

    def get_model(self):
        """Return the existing model used buy the initial learner

        """
        return self.model

    def set_model(self, model: BaseModel):
        """Set the learn model

        """
        self.model = model

    def train(self, instances):
        """Train on the set of training instances.

        Default training behavior is to use the learn model to fit data.

        Args:
            instances (List[Instance]): training instances.
  
        """
        self.model.train(instances)

    def predict(self, instances):
        """Predict classification labels for the set of instances.

        Args:
            instances (List[Instance]) or (Instance): training or test instances.
  
        Returns:
            label classifications (List(int))
  
        """
        return self.model.predict(instances)

    def set_params(self, params: Dict):
        """Set params for the initial learner.

        Defines default behavior, setting only BaseModel params

        Args:
            params (Dict): set of available params with updated values.
  
        """
        self.model.set_params(params)

    def get_available_params(self) -> Dict:
        """Get the set of params defined in the learner usage.

        Defines default behavior, return only BaseModel params.

        Returns:
            dictionary mapping param names to current values
  
        """
        return self.model.get_available_params()

    def prediction_proba(self, instances):
        return self.model.predict_proba_adversary(instances)

    def decision_function(self, instances):
        raise NotImplementedError

class ImprovedPredictor(object):
    """Base class for improved learning methods.

    Defines the bare-minimum functionality for improved learning
    strategies. Specified learning algorithms can create wrappers
    around the underlying methods.

    """

    def __init__(self):
        self.initial_learner = None   # type: InitialPredictor
        self.adversary = None         # type: AdversaryStrategy

    def improve(self, instances):
        """Improve default behavior (do nothing)

        """
        return

    def predict(self, instances):
        """Predict classification labels for the set of instances using the old model.

        Args:
            instances (List[Instance]) or (Instance): training or test instances.

        Returns:
            label classifications (List(int))

        """
        return self.initial_learner.predict(instances)

    # for future use, if learner improve strategies need more parameters
    def set_params(self, params:Dict):
        return

    def get_available_params(self) -> Dict:
        return

    def set_adversarial_params(self, learner: InitialPredictor, adversary: AdversaryStrategy):
        """Default behavior when given knowledge of initial learner and adversary.

        Args:
            learner (InitialPredictor): Learner used in initial training.
            adversary (AdversaryStrategy): Adversary used to transform instances.

        """
        self.initial_learner = learner
        self.adversary = adversary

    def predict_proba(self, instances):
        raise NotImplementedError

    def decision_function(self, instances):
        raise NotImplementedError

"""
Default learner and improved learner. Useful for wrapping
non-specialized learner models (i.e. when the adversary acts against
a non-responsive learner).
"""
class Learner(InitialPredictor):

    def __init__(self):
        InitialPredictor.__init__(self)


class ImprovedLearner(ImprovedPredictor):

    def __init__(self):
        ImprovedPredictor.__init__(self)
