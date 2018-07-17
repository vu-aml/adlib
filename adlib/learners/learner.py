from typing import Dict, List
from data_reader.binary_input import Instance


class Learner(object):
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
        self.num_features = 0
        self.training_instances = None

    def set_training_instances(self, training_data):
        """

        :param training_data: an dataset object , which when calling numpy() will return
                X: feature matrix. shape (num_instances, num_feautres_per_instance)
                y: label array. shape (num_instances, )
        """
        if isinstance(training_data, List):
            self.training_instances = training_data  # type: List[Instance]
            self.num_features = self.training_instances[0].get_feature_vector().get_feature_count()
        else:
            self.training_instances = training_data
            self.num_features = training_data.features.shape[1]

    def train(self):
        """Train on the set of training instances.

        """
        raise NotImplementedError

    def predict(self, instances):
        """Predict classification labels for the set of instances.

        Args:
            :param instances: matrix of instances shape (num_instances, num_feautres_per_instance)

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

    def predict_proba(self, X):
        """
        outputs a list of log probability of prediction
        :param X: matrix of instances shape (num_instances, num_feautres_per_instance)
        :return: list of log probability
        """
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError
