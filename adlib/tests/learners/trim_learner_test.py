# trim_learner_test.py
# Tests the TRIM learner implementation
# Matthew Sedam


from adlib.learners import SimpleLearner
from adlib.learners import TRIMLearner
from adlib.adversaries.label_flipping import LabelFlipping
from adlib.adversaries.k_insertion import KInsertion
from adlib.adversaries.datamodification.data_modification import \
    DataModification
from adlib.utils.common import calculate_correct_percentages
from adlib.utils.common import get_spam_features
from copy import deepcopy
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from sklearn import svm
import numpy as np
import sys
import time


def test_trim_learner():
    print()
    print('###################################################################')
    print('START TRIM Learner test.\n')

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
                           binary=True, raw=True)

    training_data, testing_data = dataset.split({'train': 20, 'test': 80})
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
        lnr = orig_learner.model.learner
        eye = np.eye(training_data[0].get_feature_count(), dtype=int)
        orig_theta = lnr.decision_function(eye) - lnr.intercept_[0]
        target_theta = deepcopy(orig_theta)

        spam_instances = []
        for inst in training_data + testing_data:
            if inst.get_label() == 1:
                spam_instances.append(inst)

        spam_features, ham_features = get_spam_features(spam_instances)

        # Set features to recognize spam as ham
        for index in spam_features:
            target_theta[index] = -10

        for index in ham_features:
            target_theta[index] = 0.01

        print('Features selected: ', np.array(spam_features))
        print('Number of features: ', len(spam_features))

        attacker = DataModification(orig_learner, target_theta, verbose=True)

    print('###################################################################')
    print('START', attacker_name, 'attack.\n')

    attack_data = attacker.attack(training_data)
    attack_data += testing_data
    np.random.shuffle(attack_data)

    print('\nEND', attacker_name, 'attack.')
    print('###################################################################')
    print()

    # Retrain the model with poisoned data
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, attack_data)
    learner.train()

    print('###################################################################')
    print('START TRIM learner.\n')

    # Train with TRIM learner
    trim_learner = TRIMLearner(attack_data, len(testing_data), verbose=True)
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
    print('\nTotal time: ', round(end - begin, 2), 's', '\n', sep='')

    print('\nEND TRIM Learner test.')
    print('###################################################################')
    print()


if __name__ == '__main__':
    test_trim_learner()
