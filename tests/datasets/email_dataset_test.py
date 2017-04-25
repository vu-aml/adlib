import pytest
from learners import SVMRestrained
from data_reader import input
from data_reader.input import Instance
from random import seed, shuffle
from data_reader.dataset import EmailDataset


@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/test/100_instance_debug.csv', raw=False)
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
def simple_learner(data):
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, data['training_data'])
    return learner


@pytest.fixture
def empty_learner():
    return SimpleLearner()


def bad_dataset_params1():
    with pytest.raises(AttributeError) as error:
        dataset = EmailDataset(raw=False)


def bad_dataset_params2():
    with pytest.raises(AttributeError) as error:
        dataset = EmailDataset(raw=True)


def bad_dataset_params3():
    with pytest.raises(AttributeError) as error:
        dataset = EmailDataset(path='notarealpath.pkl', features=[1, 2, 3],
                               labels=[1])

# TODO: Also test serializing then loading
def load_serialized():
    feat_val = data['training'][0].toarray()[0][0]
    assert feat_val == 1.0 or feat_val == 0.0
    label_val = data['training'][1][0]
    assert label_val == 1.0 or label_val == -1.0


def test_predict_returns_binary_label(simple_learner, testing_data):
    simple_learner.train()
    result = simple_learner.predict(testing_data[0])
    assert result in [SimpleLearner.positive_classification, SimpleLearner.negative_classification]
