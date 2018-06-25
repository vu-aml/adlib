# A set of common functions
# Matthew Sedam. 2018.

from adlib.learners import Learner
from data_reader.binary_input import Instance
from typing import List
import math
import numpy as np


def get_fvs_and_labels(instances: List[Instance]):
    """
    :param instances: the instances
    :return: the feature vector matrix and labels
    """

    fvs = []
    labels = []
    for inst in instances:
        fvs.append(np.array(inst.get_csr_matrix().todense().tolist()).flatten())
        labels.append(inst.get_label())
    fvs, labels = np.array(fvs), np.array(labels)

    return fvs, labels


def calculate_correct_percentages(orig_labels, attack_labels, instances):
    """
    Calculates the percent of labels that were predicted correctly before and
    after the attack.
    :param orig_labels: the labels predicted by the pre-attack learner
    :param attack_labels: the labels predicted by the post-attack learner
    :param instances: the list of instances
    :return: strings of original percent correct, attack percent correct, and
             the difference (original - attack)
    """

    orig_count = 0
    count = 0
    for i in range(len(instances)):
        if orig_labels[i] != instances[i].get_label():
            orig_count += 1
        if attack_labels[i] != instances[i].get_label():
            count += 1

    orig_precent_correct = ((len(instances) - orig_count) * 100
                            / len(instances))
    attack_precent_correct = ((len(instances) - count) * 100
                              / len(instances))
    difference = orig_precent_correct - attack_precent_correct

    orig_precent_correct = str(round(orig_precent_correct, 4))
    attack_precent_correct = str(round(attack_precent_correct, 4))
    difference = str(round(difference, 4))

    return orig_precent_correct, attack_precent_correct, difference


def fuzz_matrix(matrix: np.ndarray):
    """
    Add to every entry of matrix some noise to make it non-singular.
    :param matrix: the matrix - 2 dimensional
    """

    m = matrix.tolist()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            m[i][j] += abs(np.random.normal(0, 0.00001))

    return np.array(m)


def get_spam_features(instances, p=0.9):
    """
    Returns a list of feature indices where the proportion of instances that
    have them is >= p
    :param instances: the spam instances - MUST BE SPAM (i.e. have a label of 1)
    :param p: the proportion of instances that must have this value
    :return: a tuple comprised of spam and ham features in separate lists
    """

    if len(instances) == 0:
        raise ValueError('Must have at least one instance.')

    spam_features = []
    ham_features = []
    for i in range(instances[0].get_feature_count()):
        count = 0
        for inst in instances:
            count += 1 if inst.get_feature_vector().get_feature(i) > 0 else 0

        if (count / len(instances)) >= p:
            spam_features.append(i)
        else:
            ham_features.append(i)

    return spam_features, ham_features


def logistic_function(x):
    """
    :param x: x
    :return: the logistic function of x
    """

    return 1 / (1 + math.exp(-1 * x))


def logistic_loss(instances: List[Instance], lnr: Learner):
    """
    Calculates the logistic loss for instances
    :param instances: the instances
    :param lnr: the learner
    :return: the loss
    """

    fvs, labels = get_fvs_and_labels(instances)

    loss = lnr.decision_function(fvs)
    loss = list(map(lambda x, y: -1 * x * y, loss, labels))
    loss = np.array(list(map(lambda x: math.log1p(math.exp(x)), loss)))

    return loss
