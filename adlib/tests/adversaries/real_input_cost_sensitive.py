from sklearn.naive_bayes import BernoulliNB
from typing import Dict, List
import pytest
import numpy as np
from learners import learner, SimpleLearner
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance,BinaryFeatureVector
from data_reader.operations import load_dataset
from data_reader.operations import sparsify
from adversaries import CostSensitive
from copy import deepcopy



#data operation
dataset = EmailDataset(path='.data_reader/data/raw/trec05p-1/full',binary= False,raw=True)
training_, testing_ = dataset.split({'train': 60, 'test': 40})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)

#learner
learning_model = BernoulliNB()
learner = SimpleLearner(learning_model, training_data)
learner.train()

#adversary
param = {}
param['Ua'] = [[0, 20], [0, 0]]
param['Vi'] = 0
param['Uc'] = [[1 ,-1] ,[-10 ,1]]
param['scenario'] = None

adversary = CostSensitive()
adversary.set_params(param)
adversary.set_adversarial_params(learner,training_data)

#test attack
predictions1 = learner.predict(testing_data)
adversary.attack(testing_data)
predictions2 = learner.predict(testing_data)

val = [testing_data[i].label for i in range(len(testing_data))]
print(val)
print(predictions1)
print(predictions2)
