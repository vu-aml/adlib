import pytest
from adlib.learners import SVMRestrained
from data_reader import binary_input
from data_reader.binary_input import Instance
from random import seed, shuffle
from data_reader.dataset import EmailDataset

@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',binary= False,raw=True)
    # set a seed so we get the same output every time
    seed(1)
    training_data, testing_data = dataset.split({'train': 60, 'test': 40})
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
