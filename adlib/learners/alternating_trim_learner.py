# alternating_trim_learner.py
# A learner that implements the Alternating TRIM algorithm.
# Matthew Sedam

from adlib.learners import Learner, TRIMLearner
from adlib.utils.common import logistic_loss
from copy import deepcopy
from typing import Dict


class AlternatingTRIMLearner(Learner):
    """
    A learner that implements the Alternating TRIM algorithm.
    """

    def __init__(self, training_instances, max_iter=10, verbose=False):

        Learner.__init__(self)
        self.set_training_instances(deepcopy(training_instances))
        self.max_iter = max_iter
        self.verbose = verbose

        self.poison_percentage = None
        self.n = None
        self.lnr = TRIMLearner(self.training_instances, 0, verbose=self.verbose)

    def train(self):
        """
        Train on the set of training instances.
        """

        if len(self.training_instances) < 2:
            raise ValueError('Must have at least 2 instances to train.')

        step_size = 1 / len(self.training_instances)
        best_poison_percentage = 0.05
        best_lnr = None
        best_loss = None

        self.poison_percentage = 0.05
        self.n = int((1 - self.poison_percentage) *
                     len(self.training_instances))
        self.lnr.n = self.n

        while self.poison_percentage < 0.5:
            self.lnr.train()
            self.lnr.redo_problem_on_train = False

            loss = (sum(logistic_loss(self.training_instances, self.lnr)) /
                    len(self.training_instances))

            if self.verbose:
                print('\nPoison Percentage:', self.poison_percentage, '- loss:',
                      loss, '\n')

            if not best_loss or loss < best_loss:
                best_poison_percentage = self.poison_percentage
                best_loss = loss
                best_lnr = deepcopy((self.lnr.training_instances, self.lnr.n,
                                     self.lnr.lda, self.lnr.verbose, self.lnr.w,
                                     self.lnr.b))

            self.poison_percentage += step_size
            self.n = int((1 - self.poison_percentage) *
                         len(self.training_instances))
            self.lnr.n = self.n

        self.poison_percentage = best_poison_percentage
        self.n = int((1 - self.poison_percentage) *
                     len(self.training_instances))
        self.lnr = TRIMLearner(best_lnr[0], best_lnr[1], best_lnr[2], best_lnr[3])
        self.lnr.w, self.lnr.b = best_lnr[4], best_lnr[5]

    def predict(self, instances):
        return self.lnr.predict(instances)

    def set_params(self, params: Dict):
        if params['training_instances'] is not None:
            self.set_training_instances(deepcopy(params['training_instances']))
        if params['max_iter'] is not None:
            self.max_iter = params['max_iter']
        if params['verbose'] is not None:
            self.verbose = params['verbose']

        self.poison_percentage = None
        self.n = None
        self.lnr = TRIMLearner(self.training_instances, 0, verbose=self.verbose)

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        return self.lnr.decision_function(X)
