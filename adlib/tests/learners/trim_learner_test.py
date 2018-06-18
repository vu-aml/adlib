# trim_learner_test.py
# Tests the TRIM learner implementation
# Matthew Sedam


from adlib.learners import SimpleLearner
from adlib.learners import TRIM_Learner
from adlib.adversaries.label_flipping import LabelFlipping
from adlib.tests.adversaries.label_flipping_test import \
    calculate_correct_percentages
from copy import deepcopy
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from sklearn import svm
import numpy as np
import time


def test_trim_learner():
    print()
    print('###################################################################')
    print('START TRIM learner test.\n')

    begin = time.time()

    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=True, raw=True)

    training_data, testing_data = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_data)
    testing_data = load_dataset(testing_data)

    print('Training sample size: ', len(training_data), '/400\n', sep='')

    # Setting the default learner
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, training_data)
    learner.train()

    original_pred_labels = learner.predict(training_data)
    orig_learner = deepcopy(learner)

    # Execute the attack
    cost = list(np.random.binomial(2, 0.5, len(training_data)))
    total_cost = 0.3 * len(training_data)  # flip around ~30% of the labels
    attacker = LabelFlipping(learner, cost, total_cost, verbose=True)

    print('###################################################################')
    print('START label-flipping attack.\n')

    attack_data = attacker.attack(training_data)

    print('\nEND label-flipping attack.')
    print('###################################################################')
    print()

    # Retrain the model with poisoned data
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, attack_data)
    learner.train()

    print('###################################################################')
    print('START TRIM learner.\n')

    # Train with TRIM learner
    # We poisoned roughly 30% of the data, so 70% should be unpoisoned
    trim_learner = TRIM_Learner(attack_data, int(0.7 * len(attack_data)),
                                verbose=True)
    trim_learner.train()

    print('\nEND TRIM learner.')
    print('###################################################################')
    print()

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

    original_pred_labels = orig_learner.predict(testing_data)
    attack_pred_labels = learner.predict(testing_data)

    (orig_precent_correct,
     attack_precent_correct,
     difference) = calculate_correct_percentages(original_pred_labels,
                                                 attack_pred_labels,
                                                 testing_data)

    print('###################################################################')
    print('Predictions with other half of dataset:')
    print('Original correct percentage: ', orig_precent_correct, '%')
    print('Attack correct percentage: ', attack_precent_correct, '%')
    print('Difference: ', difference, '%')

    ############################################################################
    # Calculate statistics with trim learner

    data = training_data + testing_data
    trim_pred_labels = trim_learner.predict(data)
    normal_pred_labels = learner.predict(data)

    (trim_percent_correct,
     normal_percent_correct,
     difference) = calculate_correct_percentages(trim_pred_labels,
                                                 normal_pred_labels,
                                                 data)

    print('###################################################################')
    print('Predictions using TRIM learner:')
    print('TRIM learner percentage: ', trim_percent_correct, '%')
    print('Simple learner correct percentage: ', normal_percent_correct, '%')
    print('Difference: ', difference, '%')

    end = time.time()
    print('\nTotal time: ', round(begin - end, 2), 's', '\n', sep='')

    print('\nEND TRIM learner test.')
    print('###################################################################')
    print()


if __name__ == '__main__':
    test_trim_learner()
