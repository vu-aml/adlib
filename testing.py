from sklearn.naive_bayes import BernoulliNB
from typing import Dict, List
import pytest
import numpy as np
from learners import learner, SimpleLearner
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance, load_dataset
from data_reader.operations import sparsify
from adversaries import CostSensitive

dataset = EmailDataset(path='./data_reader/data/test/100_instance_debug.csv', raw=False)
training_, testing_ = dataset.split({'train': 60, 'test': 40})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)

learning_model = BernoulliNB()
learner = SimpleLearner(learning_model, training_data)
learner.train()

result = learner.predict_log_proba(testing_data[0])
print(result)