# k_insertion_test.py
# Tests the k-insertion implementation
# Matthew Sedam


from adlib.learners import SimpleLearner
from adlib.adversaries.k_insertion import KInsertion
from adlib.tests.adversaries.label_flipping_test import \
    calculate_correct_percentages
from copy import deepcopy
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from sklearn import svm
import numpy as np
import pytest
import sys


@pytest.mark.run
def test_k_insertion():
    """
    Use as follows:
    python3 adlib/tests/adversaries/k_insertion_test.py #-TO-ADD #-ITERATIONS
    """
    
    print('\n#################################################################')
    print('START k-insertion attack.\n')

    if len(sys.argv) > 2:
        number_to_add = int(sys.argv[1])
        num_iterations = int(sys.argv[2])
    else:
        number_to_add = 1
        num_iterations = 4

    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=True, raw=True)
    training_data = load_dataset(dataset)

    # Randomly choose ~12% of dataset to decrease debugging time
    # 10% was too small for a realistic calculation.
    choices = np.random.binomial(1, 0.12, len(training_data))
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
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, training_data)
    learner.train()

    original_pred_labels = learner.predict(training_data)
    before_attack_label = original_pred_labels[0]
    orig_learner = deepcopy(learner)

    # Do the attack
    attacker = KInsertion(learner,
                          training_data[0],
                          number_to_add=number_to_add,
                          num_iterations=num_iterations,
                          verbose=True)

    attack_data = attacker.attack(training_data)

    # Retrain the model with poisoned data
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, attack_data)
    learner.train()

    print('Number of added instances: ', len(attack_data) - len(training_data))
    
    ############################################################################
    # Calculate statistics with training data

    attack_pred_labels = learner.predict(training_data)  # predict w/ orig label
    after_attack_label = attack_pred_labels[0]

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

    ############################################################################
    # Calculate statistics with predict data (other half of dataset)

    print('###################################################################')
    print('Selected instance true label: ', training_data[0].get_label())
    print('Selected instance predicted label BEFORE attack: ',
          before_attack_label)
    print('Selected instance predicted label AFTER attack: ',
          after_attack_label)

    print('\nEND k-insertion attack.')
    print('#################################################################\n')


if __name__ == '__main__':
    test_k_insertion()
