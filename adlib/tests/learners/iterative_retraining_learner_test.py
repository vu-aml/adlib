# iterative_retraining_learner_test.py
# Tests the Iterative Retraining learner.
# Matthew Sedam


from adlib.adversaries.label_flipping import LabelFlipping
from adlib.adversaries.k_insertion import KInsertion
from adlib.adversaries.datamodification.data_modification import \
    DataModification
from adlib.learners import IterativeRetrainingLearner
from adlib.learners import SimpleLearner
from adlib.learners import TRIMLearner
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


def test_iterative_retraining_learner():
    print()
    print('###################################################################')
    print('START Iterative Retraining Learner test.\n')

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

    training_data, testing_data = dataset.split({'train': 50, 'test': 50})
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
        total_cost = 0.3 * len(training_data)  # flip around ~30% of the labels
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

    print('\nEND', attacker_name, 'attack.')
    print('###################################################################')
    print()

    # Retrain the model with poisoned data
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, attack_data)
    learner.train()

    print('###################################################################')
    print('START Iterative Retraining learner.\n')

    iterative_retraining_learner = IterativeRetrainingLearner(
        TRIMLearner(training_data, int(0.7 * len(training_data)), verbose=True),
        attack_data,
        verbose=True)

    iterative_retraining_learner.train()

    print('\nEND Iterative Retraining learner.')
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
    # Calculate statistics with iterative retraining learner

    data = training_data + testing_data
    iter_retrain_pred_labels = iterative_retraining_learner.predict(data)
    normal_pred_labels = learner.predict(data)

    (iter_retrain_percent_correct,
     normal_percent_correct,
     difference) = calculate_correct_percentages(iter_retrain_pred_labels,
                                                 normal_pred_labels,
                                                 data)

    print('###################################################################')
    print('Predictions using Iterative Retraining learner:')
    print('Iterative Retraining learner percentage: ',
          iter_retrain_percent_correct, '%')
    print('Simple learner correct percentage: ', normal_percent_correct, '%')
    print('Difference: ', difference, '%')

    end = time.time()
    print('\nTotal time: ', round(end - begin, 2), 's', '\n', sep='')

    print('\nEND Iterative Retraining Learner test.')
    print('###################################################################')
    print()


if __name__ == '__main__':
    test_iterative_retraining_learner()
