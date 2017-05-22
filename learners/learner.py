from typing import Dict


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
        self.num_features = 0

    def set_training_instances(self, X, y):
        """

        :param X: feature matrix. shape (num_instances, num_feautres_per_instance)
        :param y: label array. shape (num_instances, )
        :return:
        """
        self.feature_matrix = X
        self.labels = y
        self.num_features = len(X[0])

    def train(self):
        """Train on the set of training instances.

        """
        raise NotImplementedError

    def predict(self, X):
        """Predict classification labels for the set of instances.

        Args:
            :param X: matrix of instances shape (num_instances, num_feautres_per_instance)

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



