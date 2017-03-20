import pytest
from learners import SVMRestrained
from data_reader import input
from data_reader.input import Instance
from random import seed, shuffle

@pytest.fixture
def data():
    instances = input.load_instances('./data_reader/data/test/100_instance_debug')
    # set a seed so we get the same output every time
    seed(1)
    shuffle(instances)
    training_data = instances[:60]
    testing_data = instances[60:]
    return {'training_data': training_data, 'testing_data': testing_data}

@pytest.fixture
def training_data(data):
    return data['training_data']

@pytest.fixture
def testing_data(data):
    return data['testing_data']

@pytest.fixture
def restrained_learner(data):
    learner = SVMRestrained({'c_delta': 0.5}, data['training_data'])
    return learner

@pytest.fixture
def empty_learner():
    return SVMRestrained()

def test_empty_learner_constructor(empty_learner):
    assert empty_learner.bias == 0
    assert empty_learner.c_delta == 0.5
    assert empty_learner.training_instances == None
    assert empty_learner.weight_vector == None

def test_set_params(empty_learner):
    empty_learner.set_params({'c_delta': 0.7}) # non-default val
    assert empty_learner.c_delta == 0.7

def test_train_throws_error_when_no_training_instances(empty_learner):
    param_dict = {'c_delta': 0.7}
    empty_learner.set_params(param_dict)
    with pytest.raises(ValueError) as error:
        empty_learner.train()
    assert str(error.value) == 'Must set training instances before training'

def test_predict_returns_binary_label(restrained_learner, testing_data):
    restrained_learner.train()
    result = restrained_learner.predict(testing_data[0])
    assert result in [SVMRestrained.positive_classification, SVMRestrained.negative_classification]
