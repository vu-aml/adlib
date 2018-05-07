from adversaries.adversary import Adversary
from data_reader.binary_input import Instance
import copy
import cvxpy as cvx
import numpy as np
from typing import List, Dict


class LabelFlipping(Adversary):

    def __init__(self, learner, cost: List[float], total_cost: float,
                 gamma=0.05):

        Adversary.__init__(self)
        self.learner = learner
        self.cost = cost
        self.total_cost = total_cost
        self.gamma = gamma

    def attack(self, instances) -> List[Instance]:
        n = len(instances) * 2
        pred_labels = self.learner.predict(instances)
        orig_loss = []
        for i in range(len(pred_labels)):
            orig_loss.append((pred_labels[i] - instances[i].get_label()) ** 2)
        orig_loss = np.array(orig_loss + orig_loss)

        # Setup CVX

        q = cvx.Variable(n)
        w = cvx.Variable(n)
        epsilon = cvx.Variable(n)

        func = cvx.sum_entries(cvx.mul_elemwise(q, epsilon - orig_loss))
        func += n * self.gamma * (cvx.norm(w) ** 2)

        feature_vectors = []
        labels = []
        for inst in instances:
            feature_vectors.append(inst.get_feature_vector())
            labels.append(inst.get_label())
        feature_vectors = np.array(feature_vectors + feature_vectors)
        labels = np.array(labels + labels)

        cost = np.array((self.cost + self.cost)[n])

        constraints = [
            cvx.sum_entries(cvx.mul_elemwise(cost, q)) <= self.total_cost]
        for i in range(n):
            constraints.append(1 - labels[i] * cvx.sum_entries(
                cvx.mul_elemwise(w, feature_vectors[i])) <= epsilon[i])
            constraints.append(epsilon[i] >= 0)
            constraints.append(q[i] == 0 or q[i] == 1)
            if i < (n / 2):
                constraints.append(q[i] + q[i + (n / 2)] == 1)

        prob = cvx.Problem(cvx.Minimize(func), constraints)
        result = prob.solve()
        print(result)  #####################

    def set_params(self, params: Dict):
        raise NotImplementedError

    def get_available_params(self):
        raise NotImplementedError

    def set_adversarial_params(self, learner, train_instances):
        raise NotImplementedError
