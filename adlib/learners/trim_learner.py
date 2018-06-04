# trim_learner.py
# A learner that implements the TRIM algorithm described in "Manipulating
# Machine Learning- Poisoning Attacks and Countermeasures for Regression
# Learning" found at https://arxiv.org/pdf/1804.00308.pdf.
# Matthew Sedam

from data_reader.binary_input import Instance
from adlib.learners.learner import learner
from adlib.learners.simple_learner import SimpleLearner
from typing import Dict, List
import numpy as np


class TRIM_Learner(learner):
    """
    A learner that implements the TRIM algorithm described in the paper
    mentioned above.
    """

    def __init__(self, base_learner: SimpleLearner, n: int, lda=0.1):
        learner.__init__(self)
        self.base_learner = base_learner
        self.n = n
        self.lda = lda  # lambda
        self.training_instances = base_learner.training_instances
        self.num_features = base_learner.num_features

    def train(self):
        """
        Train on the set of training instances.
        """

        self.base_learner.train()

        # Create random sample of size self.n
        inst_set = []
        while len(inst_set) < self.n:
            for inst in self.training_instances:
                if np.random.binomial(1, 0.5) == 1 and len(inst_set) < self.n:
                    inst_set.append(inst)

                if len(inst_set) == self.n:
                    break

        old_loss = -1
        loss = 0
        while loss != old_loss:
            fvs, labels = TRIM_Learner.get_fv_matrix_and_labels(inst_set)
            loss_vector = self.base_learner.model.learner.decision_function(fvs)
            loss_vector -= labels
            loss_vector = list(map(lambda x: x ** 2, loss_vector))

            loss_tuples = []
            for i in range(len(loss_vector)):
                loss_tuples.append((loss_vector[i], inst_set[i]))
            loss_tuples.sort()

            inst_set = list(map(lambda tup: tup[1], loss_tuples[:self.n]))

            self.base_learner.training_instances = inst_set
            self.base_learner.train()

            old_loss = loss
            loss = self._calc_loss(inst_set)

    def _calc_loss(self, inst_set: List[Instance]):
        """
        Calculates the loss function as specified in the paper
        :param inst_set: the set of Instances
        :return: the loss
        """

        w = self.base_learner.model.learner.decision_function(
            np.eye(self.training_instances[0].get_feature_count()))
        w -= self.base_learner.model.learner.intercept_[0]

        loss = 0.5 * self.lda * (np.linalg.norm(w) ** 2)
        fvs, labels = TRIM_Learner.get_fv_matrix_and_labels(inst_set)

        # Calculate loss
        f_values = self.base_learner.model.learner.decision_function(fvs)
        tmp = np.array(list(map(lambda x: x ** 2, f_values - labels)))
        loss += (1 / self.n) * sum(tmp)

        return loss

    def predict(self, instances):
        """
        Predict classification labels for the set of instances.
        :param instances: list of Instance objects
        :return: label classifications (List(int))
        """

        raise NotImplementedError

    def set_params(self, params: Dict):
        """
        Sets parameters for the learner.
        :param params: parameters
        """

        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError

    @staticmethod
    def get_feature_vector_array(inst: Instance):
        """
        Turns the feature vector into an np.ndarray
        :param inst: the Instance
        :return: the feature vector (np.ndarray)
        """

        fv = inst.get_feature_vector()
        tmp = []
        for j in range(inst.get_feature_count()):
            if fv.get_feature(j) == 1:
                tmp.append(1)
            else:
                tmp.append(0)
        return np.array(tmp)

    @staticmethod
    def get_fv_matrix_and_labels(instances: List[Instance]):
        """
        Calculate feature vector matrix and label array
        :param instances: the list of Instances
        :return: a tuple of the feature vector matrix and labels
                 (np.ndarray, np.ndarray)
        """

        fvs = []
        labels = []
        for inst in instances:
            fvs.append(TRIM_Learner.get_feature_vector_array(inst))
            labels.append(inst.get_label())

        return np.array(fvs), np.array(labels)
