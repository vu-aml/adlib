import pytest
from adversaries import NashEq
from adversaries import ConvexLossEquilibirum, AntagonisticLossEquilibrium

from learners import NashEquilibrium
from data_reader import input
from data_reader.input import Instance
from random import seed, shuffle
import numpy as np

@pytest.fixture
def data():
    instances = input.load_instances('./data_reader/data/test/test')
    # set a seed so we get the same output every time
    seed(1)
    shuffle(instances)
    training_data = instances[:60]
    testing_data = instances[60:]
    return {'training_data': training_data, 'testing_data': testing_data}

@pytest.fixture
def training_data(data):
    return data['training_data']

@pytest.fixture()
def learner():
    return NashEquilibrium()

@pytest.fixture
def adversary():
    return NashEq()

def test_set_params(adversary):
    lambda_val = 50
    epsilon = 50
    params = {
        'game': NashEq.CONVEX_LOSS,
        'loss_function_name': 'neq_logistic_loss',
        'lambda_val': lambda_val,
        'epsilon': epsilon
    }
    adversary.set_params(params)

    assert adversary.game == NashEq.CONVEX_LOSS
    assert adversary.loss_function_name == 'neq_logistic_loss'
    assert adversary.lambda_val == lambda_val
    assert adversary.epsilon == epsilon

def test_get_available_params(adversary):
    lambda_val = 50
    epsilon = 50
    params = {
        'game': NashEq.ANTAGONISTIC_LOSS,
        'loss_function_name': 'neq_worst_case_loss',
        'lambda_val': lambda_val,
        'epsilon': epsilon
    }
    adversary.set_params(params)
    assert adversary.get_available_params() == params

def test_loss_function(adversary):
    adversary.loss_function_name = 'neq_worst_case_loss'
    assert adversary.loss_function() == adversary.neq_worst_case_loss

    adversary.loss_function_name = 'neq_linear_loss'
    assert adversary.loss_function() == adversary.neq_linear_loss

    adversary.loss_function_name = 'neq_logistic_loss'
    assert adversary.loss_function() == adversary.neq_logistic_loss

def test_set_adversarial_params_with_convex_loss(adversary, learner, training_data):
    adversary.game = NashEq.CONVEX_LOSS
    adversary.set_adversarial_params(learner, training_data)
    assert rough_compare(
        adversary.perturbation_vector,
        np.array([-2.48969869e-05,  -2.48969869e-05, -2.48969869e-05, -2.48969869e-05, -2.48969869e-05,  -2.48969869e-05]),
        8
    )
    assert rough_compare(
        adversary.learner_perturbation_vector,
        np.array([-0.00214514, -0.00214514,  0.11208145, -0.00214514, -0.00214514, 0.11208145]),
        8
    )

def test_set_adversarial_params_with_antagonistic_loss(adversary, learner, training_data):
    adversary.game = NashEq.ANTAGONISTIC_LOSS
    adversary.set_adversarial_params(learner, training_data)
    assert rough_compare(
        adversary.perturbation_vector,
        np.array([-2.68296464e-10, -2.68296439e-10, -2.68296450e-10, -2.68296444e-10, -2.68296438e-10, -2.68296435e-10]),
        8
    )
    assert rough_compare(
        adversary.learner_perturbation_vector,
        np.array([ 9.99993061e-07, 9.99986122e-07, 6.25010000e-02, 9.99979183e-07, 9.99986122e-07, 6.25009687e-02]),
        8
    )

def rough_compare(arr1, arr2, sig_figs):
    arr1 = list(map(lambda x: round(x, sig_figs), arr1))
    arr2 = list(map(lambda x: round(x, sig_figs), arr2))
    return np.array_equal(arr1, arr2)
