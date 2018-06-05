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
from sklearn import svm
import numpy as np


def test_trim_learner():
    print()
    print('###################################################################')
    print('START TRIM learner test.\n')

    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=True, raw=True)
    training_data, testing_data = dataset.split({'train': 60,
                                                 'test': 40})

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
    attacker = LabelFlipping(learner, cost, total_cost, num_iterations=2,
                             verbose=True)
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

    # We poisoned roughly 30% of the data, so 70% should be correct
    trim_learner = TRIM_Learner(training_data, int(0.7 * len(training_data)),
                                verbose=True)
    trim_learner.train()

    pred_labels = trim_learner.predict(training_data + testing_data)
    (percent_correct,
     _, _) = calculate_correct_percentages(pred_labels,
                                           pred_labels,
                                           training_data + testing_data)

    print('\nEND TRIM learner test.')
    print('###################################################################')
    print()


if __name__ == '__main__':
    test_trim_learner()
