import pytest
from adversaries import GoodWord, SimpleOptimize
from sklearn import svm
from learners import SimpleLearner
from data_reader import input
from random import shuffle

@pytest.fixture
def data():
    instances = input.load_instances('./data_reader/data/test/100_instance_debug')
    shuffle(instances)
    training_data = instances[:60]
    testing_data = instances[60:]
    return {'training_data': training_data, 'testing_data': testing_data}


@pytest.fixture
def good_word():
    return GoodWord()

@pytest.fixture
def simple_learner(data):
    learning_model = svm.SVC(probability=True, kernel='linear')
    return SimpleOptimize(learning_model, data['training_data'])

def test_default_n(good_word):
    assert good_word.n == 100

def test_set_params_n(good_word):
    good_word.set_params({'n':50})
    assert good_word.n == 50

def test_set_params_attack_type_to_best_n(good_word):
    good_word.set_params({'attack_model_type': GoodWord.BEST_N})
    assert good_word.attack_model_type == GoodWord.BEST_N

def test_set_params_attack_type_to_first_n(good_word):
    good_word.set_params({'attack_model_type': GoodWord.FIRST_N})
    assert good_word.attack_model_type == GoodWord.FIRST_N

def test_set_params_raises_exception_with_invalid_attack_type(good_word):
    with pytest.raises(ValueError) as error:
        good_word.set_params({'attack_model_type': 'Arbitrary Attack'})
    assert str(error.value) == 'Invalid attack model type'

def test_set_adversarial_params(good_word, simple_learner, data):
    good_word.set_adversarial_params(simple_learner, data['training_data'])
    assert good_word.learn_model == simple_learner
