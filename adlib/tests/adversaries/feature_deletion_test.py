import pytest
from adlib.adversaries.feature_deletion import AdversaryFeatureDeletion
from sklearn import svm
from adlib.learners import SimpleLearner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset


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
def feature_deletion(learner):
    return AdversaryFeatureDeletion(learner=learner)


def test_change_instance(feature_deletion, data):
    sample = next((x for x in data['testing_data'] if x.get_label() == 1), None)
    result = feature_deletion.change_instance(sample)
    assert sample.label == result.label


def test_set_params(feature_deletion):
    feature_deletion.set_params({'num_deletion': 50, 'all_malicious': True})
    dict = feature_deletion.get_available_params()
    assert dict['num_deletion'] == 50
    assert dict['all_malicious'] == True


def test_attack(feature_deletion, data):
    result = feature_deletion.attack(data['testing_data'])[0]
    sample = data['testing_data'][0]
    num = sample.get_feature_vector().get_feature_count()
    for i in range(num):
        assert result.get_feature_vector().get_feature(
            i) == sample.get_feature_vector().get_feature(i)


def test_attack_different(feature_deletion, data):
    feature_deletion.set_params({'num_deletion': 100, 'all_malicious': False})
    result = feature_deletion.attack(data['testing_data'])[0]
    sample = data['testing_data'][0]
    assert result.get_feature_vector().indptr[1] != sample.get_feature_vector().indptr[1]
