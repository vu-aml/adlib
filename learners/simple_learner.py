from learners.learner import RobustLearner
from learners.models import sklearner
from typing import Dict

class SimpleLearner(RobustLearner):
    """Simple Learner for initial learning methods.
    Defines the bare-minimum functionality for initial learning
    strategies.
    """

    def __init__(self, model = None, training_instances = None):
        RobustLearner.__init__(self)
        if model: self.set_model(model)
        else: self.model = None
        self.training_instances = training_instances

    def set_model(self, model):
        self.model = sklearner.Model(model)

    def train(self):
        if not self.model:
            raise ValueError('Must specify classification model')
        if not self.training_instances:
            raise ValueError('Must set training instances before training')
        self.model.train(self.training_instances)

    def predict(self, instances):
        return self.model.predict(instances)
    
    # should we also set the model params?
    def set_params(self, params: Dict):
        if params['model'] is not None:
            self.model = self.set_model(params['model'])
