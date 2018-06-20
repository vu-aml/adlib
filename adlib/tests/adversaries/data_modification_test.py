# data_modification_test.py
# Tests the data modification implementation
# Matthew Sedam


from adlib.learners import SimpleLearner
from adlib.adversaries.datamodification.data_modification import \
    DataModification
from adlib.utils.common import calculate_correct_percentages
from adlib.utils.common import get_spam_features
from copy import deepcopy
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from sklearn import svm
import numpy as np
import time


def test_data_modification():
    print()
    print('###################################################################')
    print('START data modification attack.\n')

    begin = time.time()

    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=False, raw=True)

    training_data, predict_data = dataset.split({'train': 50, 'test': 50})
    training_data = load_dataset(training_data)
    predict_data = load_dataset(predict_data)

    print('Training sample size: ', len(training_data), '/400\n', sep='')

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

    target_theta = calculate_target_theta(orig_learner,
                                          training_data,
                                          predict_data)

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

    ############################################################################
    # Calculate statistics with predict data (other half of dataset)

    spam_pred_labels = learner.predict(spam_instances)
    spam_ham_count = sum(map(lambda x: 1 if x == -1 else 0, spam_pred_labels))
    print('###################################################################')
    print('Number of spam instances in original training set that were \n',
          'classified as ham after the attack: ', spam_ham_count, '/',
          len(spam_instances), sep='')

    end = time.time()
    print('\nTotal time: ', round(end - begin, 2), 's', '\n', sep='')

    print('\nEND data modification attack.')
    print('###################################################################')
    print()


def calculate_target_theta(orig_learner, training_data, predict_data):
    lnr = orig_learner.model.learner
    eye = np.eye(training_data[0].get_feature_count(), dtype=int)
    orig_theta = lnr.decision_function(eye) - lnr.intercept_[0]
    target_theta = deepcopy(orig_theta)

    spam_instances = []
    for inst in training_data + predict_data:
        if inst.get_label() == 1:
            spam_instances.append(inst)

    spam_features, ham_features = get_spam_features(spam_instances)

    # Set features to recognize spam as ham
    for index in spam_features:
        target_theta[index] = -1

    print('Features selected: ', np.array(spam_features))
    print('Number of features: ', len(spam_features))

    return target_theta


if __name__ == '__main__':
    test_data_modification()
