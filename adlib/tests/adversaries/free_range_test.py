import pytest
from typing import Dict,List
from adversaries import FreeRange
from sklearn import svm
from adlib.learners import learner, SimpleLearner,SVMFreeRange
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance
from data_reader.operations import load_dataset

@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/test/100_instance_debug.csv', raw=False)
    training_, testing_ = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)
    return {'training_data': training_data, 'testing_data': testing_data}

@pytest.fixture
def training_data(data):
    return data['training_data']

@pytest.fixture
def testing_data(data):
    return data['testing_data']

@pytest.fixture
def free_range():
    return FreeRange()

@pytest.fixture
def freerange_learner(data):
    return SVMFreeRange({'c_f': 0.7, 'xmin': 0.25, 'xmax': 0.75}, \
                        data['training_data'])


def test_set_adversarial_params(free_range,freerange_learner,training_data):
    free_range.set_adversarial_params(freerange_learner,training_data)
    assert free_range.learn_model == freerange_learner
    #may need to test num_features and innocuous value chosen

def test_transform_instance(free_range,freerange_learner,training_data,testing_data):
    free_range.set_adversarial_params(freerange_learner,training_data)
    #if f_attack is 1, the result should be exactly the same as innocuous target
    param = {}
    param['f_attack'] = 1
    free_range.set_params(param)
    sample_ = next((x for x in testing_data if x.get_label() == learner.positive_classification),None)
    free_range.transform(sample_)
    for i in range(0, free_range.num_features):
        delta = free_range.innocuous_target.get_feature_vector().get_feature(i) \
                   - sample_.get_feature_vector().get_feature(i)
        assert delta == 0

def test_transform_instance_low(free_range, freerange_learner, training_data, testing_data):
    free_range.set_adversarial_params(freerange_learner, training_data)
    #if f_attack is low, the result is different from the innocuous target
    param = {}
    param['f_attack'] = 0.01
    sample = next((x for x in testing_data if x.get_label() == learner.positive_classification),None)
    free_range.transform(sample)
    equal = True
    for i in range(0, free_range.num_features):
        delta = free_range.innocuous_target.get_feature_vector().get_feature(i) \
                   -sample.get_feature_vector().get_feature(i)
        if delta != 0:
            equal = False
    assert not equal
