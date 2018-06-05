from sklearn.naive_bayes import GaussianNB
import pytest
from adlib.learners import learner, SimpleLearner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adlib.adversaries import CostSensitive
from copy import deepcopy


@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/full', binary=False, raw=True)
    training_, testing_ = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)
    return {'training_data': training_data, 'testing_data': testing_data}


@pytest.fixture
# set the parameters according to the experiments provided
# UA = [[0,20],[0,0]]
# Uc = [[1,-1],[-10,1]]
# Vi = 0
def cost_sensitive():
    adversary = CostSensitive(binary=False)
    param = {}
    param['Ua'] = [[0, 20], [0, 0]]
    param['Vi'] = 0
    param['Uc'] = [[1, -1], [-10, 1]]
    param['scenario'] = None
    adversary.set_params(param)
    return adversary


@pytest.fixture
def NB_learner(data):
    learner_model = GaussianNB()
    learner = SimpleLearner(model=learner_model, training_instances=data['training_data'])
    learner.train()
    return learner


def test_set_adversarial_params(cost_sensitive, NB_learner, data):
    cost_sensitive.set_adversarial_params(NB_learner, data['training_data'])
    assert cost_sensitive.learn_model == NB_learner
    assert cost_sensitive.delta_Ua == 20


def test_log_threshold(cost_sensitive):
    assert cost_sensitive.log_threshold() == 5.5
    assert cost_sensitive.log_threshold([[1, -1], [-1, 1]]) == 1


def test_gap_negative_instances(cost_sensitive, NB_learner, data):
    cost_sensitive.set_adversarial_params(NB_learner, data['training_data'])
    sample = next((x for x in data['testing_data'] if x.get_label() ==
                   learner.negative_classification), None)
    # gap(x) <= 0 for all negative_classified instances
    result = cost_sensitive.gap(sample)
    assert result <= 0


def test_find_MCC(cost_sensitive, NB_learner, data):
    cost_sensitive.set_adversarial_params(NB_learner, data['training_data'])
    sample = next((x for x in data['testing_data'] if x.get_label() ==
                   learner.negative_classification), None)
    x_cost, x_list = cost_sensitive.find_mcc(cost_sensitive.num_features,
                                             cost_sensitive.gap(sample), sample)
    y_cost, y_list = cost_sensitive.find_mcc(0, cost_sensitive.gap(sample), sample)
    assert x_cost == 0
    assert y_list == []
    assert x_list == []


# this runs longer than expected.
def test_A_x_(cost_sensitive, NB_learner, data):
    cost_sensitive.set_adversarial_params(NB_learner, data['training_data'])
    sample = next((x for x in data['testing_data'] if x.get_label() ==
                   learner.positive_classification), None)
    result = deepcopy(sample)
    cost_sensitive.a(result)
    assert result.label == sample.label
