import pytest
from learners.models import sklearner
from sklearn import svm
from learners import SimpleLearner
from data_reader import binary_input
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance
from data_reader.binary_input import load_dataset
from random import seed, shuffle

@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/test/100_instance_debug.csv', raw=False)
    # set a seed so we get the same output every time
    seed(1)
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
def simple_learner(data):
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, data['training_data'])
    return learner

@pytest.fixture
def empty_learner():
    return SimpleLearner()

def test_empty_learner_constructor(empty_learner):
    assert empty_learner.model == None
    assert empty_learner.training_instances == None

def test_set_model_sets_sklearner_object(empty_learner):
    learning_model = svm.SVC(probability=True, kernel='linear')
    empty_learner.set_model(learning_model)
    assert isinstance(empty_learner.model, sklearner.Model)

def test_train_throws_error_when_no_model(empty_learner):
    with pytest.raises(ValueError) as error:
        empty_learner.train()
    assert str(error.value) == 'Must specify classification model'

def test_train_throws_error_when_no_training_instances(empty_learner):
    learning_model = svm.SVC(probability=True, kernel='linear')
    empty_learner.set_model(learning_model)
    with pytest.raises(ValueError) as error:
        empty_learner.train()
    assert str(error.value) == 'Must set training instances before training'

def test_predict_returns_binary_label(simple_learner, testing_data):
    simple_learner.train()
    sample_ = testing_data[0]
    result = simple_learner.predict(sample_)
    assert result in [SimpleLearner.positive_classification, SimpleLearner.negative_classification]
