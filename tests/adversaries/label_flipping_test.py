# label_flipping_test.py
# Tests the label flipping implementation

from sklearn import svm
from learners import SimpleLearner
import numpy as np
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.label_flipping import LabelFlipping


def test_label_flipping():
    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='../../data_reader/data/raw/trec05p-1/test-400',
                           binary=True, raw=True)
    training_, testing_ = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)

    # Setting the default learner
    # Test simple learner svm
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, training_data)
    learner.train()

    # Execute the attack
    cost = list(np.full(len(training_data), 1))  # cost of flip is constant 1
    total_cost = 0.1 * len(training_data)  # flip at most 10% of labels
    attacker = LabelFlipping(learner, cost, total_cost)
    result = attacker.attack(training_data)
    print(result)


if __name__ == '__main__':
    test_label_flipping()
