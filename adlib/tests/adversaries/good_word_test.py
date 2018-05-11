import pytest
from adversaries import GoodWord
from sklearn import svm
from learners import learner, SimpleLearner
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance
from data_reader.operations import load_dataset

@pytest.fixture
def data():
    dataset = EmailDataset(path='./data_reader/data/test/100_instance_debug.csv', raw=False)
    training_, testing_ = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)
    return {'training_data': training_data, 'testing_data': testing_data}

@pytest.fixture
def training_data(data):
    return data['training_data']

@pytest.fixture
def good_word():
    return GoodWord()

@pytest.fixture
def simple_learner(data):
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, data['training_data'])
    learner.train()
    return learner

@pytest.fixture()
def good_word_with_params(simple_learner, training_data):
    adv = GoodWord()
    adv.set_adversarial_params(simple_learner, training_data)
    return adv

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
    assert good_word.positive_instance.get_label() == learner.positive_classification
    assert good_word.negative_instance.get_label() == learner.negative_classification
    # add feature space test

# assumes data has at least one instance that has at least one index with instance[index] = 0
def test_add_words_to_instance_flips_bit_for_new_word(good_word, training_data):
    feature_vector = training_data[0].get_feature_vector()
    # gets first zero index since feature vector indices only contain indices of nonzero elements
    zero_val_index = next((index for index in range(len(feature_vector)+1) if feature_vector[index] != index), None)
    assert feature_vector.get_feature(zero_val_index) == 0
    good_word.add_words_to_instance(training_data[0], [zero_val_index])
    assert feature_vector.get_feature(zero_val_index) == 1

# assumes training_data has at least one instance that has at least one index with instance[index] = 1
def test_add_words_to_instance_no_change_for_existing_word(good_word, training_data):
    feature_vector = training_data[0].get_feature_vector()
    one_val_index = feature_vector[0]
    assert feature_vector.get_feature(one_val_index) == 1
    good_word.add_words_to_instance(training_data[0], [one_val_index])
    assert feature_vector.get_feature(one_val_index) == 1

def test_find_witness_returns_messages_differing_by_one_word(good_word_with_params):
    adv = good_word_with_params
    spam_msg, legit_msg = adv.find_witness()
    assert adv.predict(Instance(0, legit_msg)) == learner.negative_classification
    assert adv.predict(Instance(0, spam_msg)) == learner.positive_classification
    assert len(legit_msg.feature_difference(spam_msg)) == 1

def test_get_n_words_with_first_n_model(good_word_with_params):
    adv = good_word_with_params
    num_words = 15
    adv.set_params({'n': num_words, 'attack_model_type': GoodWord.FIRST_N})
    words = adv.get_n_words()
    # use mock to assert first_n gets called
    # may throw errors when words found are less than num_words
    assert len(words) <= num_words

def test_get_n_words_with_best_n_model(good_word_with_params):
    adv = good_word_with_params
    num_words = 15
    adv.set_params({'n': num_words, 'attack_model_type': GoodWord.BEST_N})
    words = adv.get_n_words()
    # use mock to assert best_n gets called
    assert len(words) <= num_words
