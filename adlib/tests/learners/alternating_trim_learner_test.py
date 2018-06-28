# alternating_trim_learner_test.py
# Tests the Alternating TRIM learner implementation
# Matthew Sedam


from adlib.learners import SimpleLearner
from adlib.learners import AlternatingTRIMLearner
from adlib.adversaries.label_flipping import LabelFlipping
from adlib.adversaries.k_insertion import KInsertion
from adlib.adversaries.datamodification.data_modification import \
    DataModification
from adlib.tests.adversaries.data_modification_test import \
    calculate_target_theta
from adlib.utils.common import calculate_correct_percentages
from copy import deepcopy
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from sklearn import svm
import numpy as np
import sys
import time


# TODO: Switch attack back


def test_alternating_trim_learner():
    print()
    print('###################################################################')
    print('START Alternating TRIM Learner test.\n')

    begin = time.time()

    if len(sys.argv) == 2 and sys.argv[1] in ['label-flipping',
                                              'k-insertion',
                                              'data-modification']:
        attacker_name = sys.argv[1]
    else:
        attacker_name = 'label-flipping'

    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=False, raw=True)

    training_data, testing_data = dataset.split({'train': 25, 'test': 75})
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
    if attacker_name == 'label-flipping':
        cost = list(np.random.binomial(2, 0.5, len(training_data)))
        total_cost = 0.25 * len(training_data)  # flip around ~25% of the labels
        attacker = LabelFlipping(learner, cost, total_cost, verbose=True)
    elif attacker_name == 'k-insertion':
        number_to_add = int(0.25 * len(training_data))
        attacker = KInsertion(learner,
                              training_data[0],
                              number_to_add=number_to_add,
                              verbose=True)
    else:  # attacker_name == 'data-modification'
        target_theta = calculate_target_theta(orig_learner,
                                              training_data,
                                              testing_data)

        attacker = DataModification(orig_learner, target_theta, verbose=True)

    print('###################################################################')
    print('START', attacker_name, 'attack.\n')

    attack_data = attacker.attack(training_data)
    # attack_data = deepcopy(training_data)
    # tmp = np.random.binomial(1, 0.25, len(training_data))
    # for i, val in enumerate(tmp):
    #     if val == 1:
    #         attack_data[i].set_label(-1 * attack_data[i].get_label())

    print('\nEND', attacker_name, 'attack.')
    print('###################################################################')
    print()

    # Retrain the model with poisoned data
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, attack_data)
    learner.train()

    print('###################################################################')
    print('START Alternating TRIM learner.\n')

    # Train with TRIM learner
    alt_trim_learner = AlternatingTRIMLearner(training_data, verbose=True)
    alt_trim_learner.train()

    print('\nEND Alternating TRIM learner.')
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
    trim_pred_labels = alt_trim_learner.predict(data)
    normal_pred_labels = learner.predict(data)

    (trim_percent_correct,
     normal_percent_correct,
     difference) = calculate_correct_percentages(trim_pred_labels,
                                                 normal_pred_labels,
                                                 data)

    print('###################################################################')
    print('Predictions using Alternating TRIM learner:')
    print('Alternating TRIM learner percentage: ', trim_percent_correct, '%')
    print('Simple learner correct percentage: ', normal_percent_correct, '%')
    print('Difference: ', difference, '%')

    end = time.time()
    print('\nTotal time: ', round(end - begin, 2), 's', '\n', sep='')

    print('\nEND Alternating TRIM learner test.')
    print('###################################################################')
    print()


if __name__ == '__main__':
    test_alternating_trim_learner()
