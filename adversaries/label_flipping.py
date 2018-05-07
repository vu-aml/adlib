from adversaries.adversary import Adversary
from data_reader.binary_input import Instance
import copy
import cvxpy as cvx
import numpy as np
from typing import List, Dict


class LabelFlipping(Adversary):

    def __init__(self, learner, train_instances: List[Instance],
                 cost: List[float], total_cost: float, gamma=0.05):

        if len(cost) != len(train_instances):
            raise ValueError('cost must have same length as train_instances')

        Adversary.__init__(self)
        self.learner = learner
        self.orig_train_instances = train_instances
        self.train_instances = train_instances
        temp = copy.deepcopy(train_instances)
        for i in temp:
            i.label = -i.label
        self.train_instances += temp
        self.cost = np.append(np.zeros(len(train_instances)), np.array(cost))
        self.total_cost = total_cost
        self.gamma = gamma

    def attack(self, instances) -> List[Instance]:
        pred_labels = self.learner.predict(self.orig_train_instances)
        orig_loss = []
        for i in range(len(pred_labels)):
            orig_loss.append((pred_labels[i] -
                              self.orig_train_instances[i].get_label()) ** 2)
        orig_loss = np.array(orig_loss + orig_loss)

        # Setup CVX

        q = cvx.Variable(len(self.train_instances))
        w = cvx.Variable(len(self.train_instances))
        epsilon = cvx.Variable(len(self.train_instances))

        func = cvx.sum_entries(cvx.mul_elemwise(q, epsilon - orig_loss))
        func += len(self.train_instances) * self.gamma * (cvx.norm(w) ** 2)

        feature_vectors = []
        for inst in self.train_instances:
            feature_vectors.append(inst.get_feature_vector())
        feature_vectors = np.array(feature_vectors)

        labels = []
        for inst in self.train_instances:
            labels.append(inst.get_label())
        labels = np.array(labels)

        constraints = [
            cvx.sum_entries(cvx.mul_elemwise(self.cost, q)) <= self.total_cost]
        for i in range(len(self.train_instances)):
            constraints.append(1 - labels[i] * cvx.sum_entries(
                cvx.mul_elemwise(w, feature_vectors[i])) <= epsilon[i])
            constraints.append(epsilon[i] >= 0)
            constraints.append(q[i] == 0 or q[i] == 1)
            if i < len(self.orig_train_instances):
                constraints.append(
                    q[i] + q[i + len(self.orig_train_instances)] == 1)

        prob = cvx.Problem(cvx.Minimize(func), constraints)
        result = prob.solve()
        print(result)  #####################

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
