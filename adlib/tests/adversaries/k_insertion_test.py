# k_insertion_test.py
# Tests the k-insertion implementation
# Matthew Sedam

from sklearn import svm
from adlib.learners import SimpleLearner
import numpy as np
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adlib.adversaries.k_insertion import KInsertion


def test_k_insertion():
    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=True, raw=True)
    training_data = load_dataset(dataset)

    # Setting the default learner
    # Test simple learner svm
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, training_data)
    learner.train()

    attacker = KInsertion(learner, training_data[0])
    attacker.attack(training_data)
