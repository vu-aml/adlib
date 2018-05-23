# data_modification_test.py
# Tests the data modification implementation
# Matthew Sedam


from adlib.learners import SimpleLearner
from adlib.adversaries.data_modification import DataModification
from adlib.tests.adversaries.label_flipping_test import \
    calculate_correct_percentages
from copy import deepcopy
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from sklearn import svm
import numpy as np
import pytest


@pytest.mark.run
def test_data_modification():
    print('\n#################################################################')
    print('START data modification attack.\n')

    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=True, raw=True)
    training_data = load_dataset(dataset)

    # Randomly choose ~15% of dataset to decrease debugging time
    choices = np.random.binomial(1, 0.15, len(training_data))
    temp = []
    predict_data = []
    count = 0
    for i in range(len(training_data)):
        if choices[i] == 1:
            temp.append(training_data[i])
            count += 1
        else:
            predict_data.append(training_data[i])
    training_data = temp
    print('Training sample size: ', count, '/400\n', sep='')

    # Setting the default learner
    # Test simple learner svm
    orig_learning_model = svm.SVC(probability=True, kernel='linear')
    orig_learner = SimpleLearner(orig_learning_model, training_data)
    orig_learner.train()

    ############################################################################
    # Calculate target theta, 1 -> spam, -1 -> ham; For the target theta
    # calculation, I am assuming I know which spam I want to be classified
    # as ham and the features I want to have a disproportionate effect on the
    # decision function calculation. For example, if feature #32 is something
    # that all of my spam has in common, I want to make the entry corresponding
    # to #32 (index 32 - 1 = 31) in target_theta to be disproportionately
    # negative so that when my spam is being classified, the 1 indicating that
    # feature #32 is present will be multiplied by a large negative number so as
    # to decrease the value of the decision function and hopefully make it
    # negative so as to classify my spam as ham.

    lnr = orig_learner.model.learner
    eye = np.eye(training_data[0].get_feature_count(), dtype=int)
    orig_theta = lnr.decision_function(eye) - lnr.intercept_[0]
    target_theta = deepcopy(orig_theta)

    spam_instances = []
    for inst in training_data:
        if inst.get_label() == 1:
            spam_instances.append(inst)

    features = get_spam_features(spam_instances)
    mean = np.mean(orig_theta)
    std = np.std(orig_theta)

    # Set features to mean + 3 STD and make it negative to help it be classified
    # as ham and NOT spam
    value = -1 * (mean + 3 * std)

    for index in features:
        target_theta[index] = value

    ############################################################################

    # Get original predictions
    original_pred_labels = orig_learner.predict(training_data)

    # Do the attack
    attacker = DataModification(orig_learner, target_theta, verbose=True)
    attack_data = attacker.attack(training_data)

    # Retrain the model with poisoned data
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, attack_data)
    learner.train()

    ############################################################################
    # Calculate statistics with training data

    attack_pred_labels = learner.predict(training_data)  # predict w/ orig label

    (orig_precent_correct,
     attack_precent_correct,
     difference) = calculate_correct_percentages(original_pred_labels,
                                                 attack_pred_labels,
                                                 training_data)

    print('###################################################################')
    print('Predictions with training dataset:')
    print('Original correct percentage: ', orig_precent_correct, '%')
    print('Attack correct percentage: ', attack_precent_correct, '%')
    print('Difference: ', difference, '%')

    ############################################################################
    # Calculate statistics with predict data (other half of dataset)

    original_pred_labels = orig_learner.predict(predict_data)
    attack_pred_labels = learner.predict(predict_data)

    (orig_precent_correct,
     attack_precent_correct,
     difference) = calculate_correct_percentages(original_pred_labels,
                                                 attack_pred_labels,
                                                 predict_data)

    print('###################################################################')
    print('Predictions with other half of dataset:')
    print('Original correct percentage: ', orig_precent_correct, '%')
    print('Attack correct percentage: ', attack_precent_correct, '%')
    print('Difference: ', difference, '%')

    print('\nEND data modification attack.')
    print('#################################################################\n')


def get_spam_features(instances, p=0.75):
    """
    Returns a list of feature indices where the proportion of instances that
    have them is >= p
    :param instances: the spam instances - MUST BE SPAM (i.e. have a label of 1)
    :param p: the proportion of instances that must have this value
    :return: the list of feature indices
    """

    if len(instances) == 0:
        raise ValueError('Must have at least one instance.')

    features = []
    for i in range(instances[0].get_feature_count()):
        count = 0
        for inst in instances:
            count += 1 if inst.get_feature_vector().get_feature(i) == 1 else 0

        if (count / len(instances)) >= p:
            features.append(i)

    return features


if __name__ == '__main__':
    test_data_modification()
