from learners.learner import RobustLearner
from typing import Dict, List
import numpy as np
from data_reader.input import Instance
from adversaries.stackelberg_new2 import SPG ,LogisticLoss
from data_reader.operations import sparsify

"""Companion class to adversary Stackelberg equilibrium.

Concept:
    Learner trains (in the usual way) and defines its loss function, costs, and
    density weight. Improved learner gathers equilibrium solution weight vector
    (as determined by adversary) and uses it to predict on the transformed data.

"""


class StackelbergLearner(RobustLearner):

    def __init__(self, params:Dict=None, training_instances=None):
        RobustLearner.__init__(self)
        self.learner_cost_array = None # type: List[float]
        self.attacker_cost_array = None  # type: List[float]
        self.learner_regularization_param = 0.2
        self.attacker_regularization_param = 0.2
        self.loss_type = SPG.LOGISTIC_LOSS
        self.game = None
        self.trained_weight = None
        if params is not None:
            self.set_params(params)
        if training_instances is not None:
            self.set_training_instances(training_instances)



    def set_params(self, params:Dict):
        if 'loss_type' in params:
            self.loss_type = params['loss_type']
        if 'learner_costs' in params:
            self.learner_cost_array = params['learner_costs']
        if 'learner_regularization' in params:
            self.density_weight = params['learner_regularization']
        if 'attacker_costs' in params:
            self.attacker_cost_array = params['attacker_costs']
        if 'attacker_regularization' in params:
            self.attacker_regularization_param = params['attacker_regularization']

    def train(self):

        # if cost not specified, use default value 1/N, N= num_instances
        if self.learner_cost_array is None:
            self.learner_cost_array = np.array([1/len(self.training_instances)]*len(self.training_instances))
        if self.attacker_cost_array is None:
            self.attacker_cost_array= np.array([1/len(self.training_instances)]*len(self.training_instances))

        if self.loss_type == SPG.LOGISTIC_LOSS:
            self.game = LogisticLoss(self.attacker_cost_array, self.learner_cost_array,
                                     self.attacker_regularization_param, self.learner_regularization_param,
                                     self.training_instances)
        else:
            raise NotImplementedError

        self.game.optimize()
        self.trained_weight = self.game.weight_vector

        print('training completed')

    def predict(self, instances):
        predictions = []
        for instance in instances:
            features = instance.get_feature_vector().get_csr_matrix().toarray()
            predictions.append(np.sign(self.predict_instance(features)))
        return predictions

    def predict_instance(self, features):
        '''
        predict class for a single instance and return a real value
        :param features: np.array of shape (1, self.num_features), i.e. [[1, 2, ...]]
        :return: float
        '''

        return self.trained_weight.dot(features.T)[0][0]

    def predict_proba(self, instances: List[Instance]):
        return [self.predict_instance(
            ins.get_feature_vector().get_csr_matrix().toarray()) for ins in instances]

    def decision_function(self):
        return self.trained_weight
