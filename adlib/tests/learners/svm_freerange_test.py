import pytest
from adlib.learners import SVMFreeRange
from random import seed
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset


@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400', binary=False, raw=True)
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
def freerange_learner(data):
    return SVMFreeRange({'c_f': 0.7, 'xmin': 0.25, 'xmax': 0.75}, \
                        data['training_data'])


@pytest.fixture
def empty_learner():
    return SVMFreeRange()


def test_empty_learner_constructor(empty_learner):
    assert empty_learner.bias == 0.0
    assert empty_learner.xmin == 0.0
    assert empty_learner.xmax == 1.0
    assert empty_learner.c_f == 0.5
    assert empty_learner.training_instances == None
    assert empty_learner.weight_vector == None


def test_set_params(empty_learner):
    empty_learner.set_params({'c_f': 0.7,  # non-default vals
                              'xmin': 0.25,
                              'xmax': 0.75})
    assert empty_learner.c_f == 0.7
    assert empty_learner.xmin == 0.25
    assert empty_learner.xmax == 0.75


def test_train_throws_error_when_no_training_instances(empty_learner):
    param_dict = {'c_f': 0.7, 'xmin': 0.25, 'xmax': 0.75}
    empty_learner.set_params(param_dict)
    with pytest.raises(ValueError) as error:
        empty_learner.train()
    assert str(error.value) == 'Must set training instances before training'


def test_predict_returns_binary_label(freerange_learner, testing_data):
    freerange_learner.train()
    sample_ = testing_data[0]
    result = freerange_learner.predict(sample_)
    assert result in [SVMFreeRange.positive_classification, SVMFreeRange.negative_classification]
