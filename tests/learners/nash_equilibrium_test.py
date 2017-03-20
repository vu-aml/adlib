import pytest
from learners import NashEquilibrium
from data_reader import input
from data_reader.input import Instance
from random import seed, shuffle
import numpy as np

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
def nash_eq_learner():
    return NashEquilibrium()

# adversary to return arbitrary weight vector
class DummyAdversary:

    def __init__(self, vec_length):
        self.vec_length = vec_length

    def get_learner_params(self):
        return [1 for i in range(self.vec_length)]

def test_constructor_initializes_empty_weight_vector(nash_eq_learner):
    assert nash_eq_learner.weight_vector == None

def test_loss_fcn_is_callable(nash_eq_learner):
    assert callable(nash_eq_learner.loss_function)

def test_get_loss_fcn_returns_loss_fcn(nash_eq_learner):
    assert nash_eq_learner.get_loss_function() == nash_eq_learner.loss_function

def test_learner_set_params(nash_eq_learner):
    adv = DummyAdversary(10) # arbitrary vec length used, this will cause an error if predict is called
    nash_eq_learner.set_params({'adversary': adv})
    assert nash_eq_learner.weight_vector == adv.get_learner_params()

def test_predict_returns_binary_label(nash_eq_learner, testing_data):
    adv = DummyAdversary(testing_data[0].get_feature_vector().feature_count)
    nash_eq_learner.set_params({'adversary': adv})

    # check if results contains only binary labels with set difference
    results = nash_eq_learner.predict(testing_data)
    results = np.array(results)
    allowed_vals = np.array([
        NashEquilibrium.positive_classification,
        NashEquilibrium.negative_classification
    ])
    diff = np.setdiff1d(results, allowed_vals)
    assert len(diff) == 0

def test_predict_raises_error_when_weight_vector_is_none(nash_eq_learner, testing_data):
    with pytest.raises(ValueError) as error:
        nash_eq_learner.predict(testing_data)
    assert str(error.value) == 'Weight vector cannot be None'
