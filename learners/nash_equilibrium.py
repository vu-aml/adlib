from learners.learner import RobustLearner
from typing import List, Dict
from types import FunctionType
import numpy as np
from util import predict_instance
from data_reader.input import Instance, FeatureVector


"""Companion class to adversary Nash Equilibrium.

Concept:
    Learner trains (in the usual way) and defines its loss function.
    Improved learner gathers equilibrium solution weight vector (as determined
    by adversary) and uses it to predict on the transformed data.

"""

class NashEquilibrium(RobustLearner):
    def __init__(self):
        """New generic initial learner with no specified learning model.

        """
        RobustLearner.__init__(self)
        self.weight_vector = None

    def predict(self, instances):
        """Predict classification labels for the set of instances.

        Args:
            instances (List[Instance]) or (Instance): training or test instances.

        Returns:
            label classifications (List(int))

        """
        if not self.weight_vector:
            raise ValueError('Weight vector cannot be None')

        predictions = []
        for instance in instances:
            features = instance.get_feature_vector().get_csr_matrix().toarray()[0]
            predictions.append(np.sign(predict_instance(features, self.weight_vector)))
        return predictions

    def set_params(self, params: Dict):
        """Set params for the initial learner.

        Defines default behavior, setting only BaseModel params

        Args:
            params (Dict): set of available params with updated values.

        """
        if params['adversary'] is not None:
            self.weight_vector = params['adversary'].get_learner_params()

    # not sure we need these
    def loss_function(self, z: float, y: int):
        return np.log(1 + np.exp(-1 * y * z))

    def get_loss_function(self) -> FunctionType:
        return getattr(self, 'loss_function')
