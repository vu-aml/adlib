from typing import List, Dict
from data_reader.input import Instance


class RobustLearner(object):
    """Base class for initial learning methods.

    Defines the bare-minimum functionality for initial learning
    strategies. Specified learning algorithms can create wrappers
    around the underlying methods.

    """
    positive_classification = 1
    negative_classification = -1

    def __init__(self):
        """New generic initial learner with no specified learning model.

        """
        self.training_instances = None
        self.num_features = 0

    def set_training_instances(self, training_instances):
        self.training_instances = training_instances  # type: List[Instance]
        self.num_features = self.training_instances[0].get_feature_vector().get_feature_count()

    def train(self):
        """Train on the set of training instances.

        """
        raise NotImplementedError

    def predict(self, instances):
        """Predict classification labels for the set of instances.

        Args:
            instances (List[Instance]) or (Instance): training or test instances.

        Returns:
            label classifications (List(int))

        """
        raise NotImplementedError

    def set_params(self, params: Dict):
        """Set params for the initial learner.

        Defines default behavior, setting only BaseModel params

        Args:
            params (Dict): set of available params with updated values.

        """
        raise NotImplementedError

    def predict_proba(self, instances):
        raise NotImplementedError

    def decision_function(self, instances):
        raise NotImplementedError


