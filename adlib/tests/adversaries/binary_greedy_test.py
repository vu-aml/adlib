from sklearn import svm
import pytest
from adlib.learners import learner, SimpleLearner
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance,BinaryFeatureVector
from data_reader.operations import load_dataset
from data_reader.operations import sparsify
from adversaries.binary_greedy import BinaryGreedy

@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/test/100_instance_debug.csv', raw=False)
    training_, testing_ = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)
    return {'training_data': training_data, 'testing_data': testing_data}

@pytest.fixture
def learner(data):
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, data['training_data'])
    learner.train()
    return learner

@pytest.fixture
def binary_greedy(data,learner):
    return BinaryGreedy(learner= learner)

def test_default_setting(binary_greedy):
    assert binary_greedy.lambda_val == 0.05
    assert binary_greedy.epsilon == 0.0002
    assert binary_greedy.step_size == 0.05
    assert binary_greedy.weight_vector is not None

def test_set_adversarial_params(binary_greedy,learner,data):
    binary_greedy.set_adversarial_params(learner = learner, train_instances= data['training_data'])
    assert binary_greedy.num_features != 0

def test_attack(binary_greedy,data ):
    result = binary_greedy.attack(data['testing_data'])
    assert len(result) == len(data['testing_data'])
    #should write more tests





