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

    # Randomly choose ~40% of dataset to decrease debugging time
    choices = np.random.binomial(1, 0.4, len(training_data))
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

    # Get original predictions
    original_pred_labels = orig_learner.predict(training_data)

    # Do the attack
    attacker = DataModification(orig_learner, True)
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


if __name__ == '__main__':
    test_data_modification()
