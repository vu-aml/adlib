from adlib.adversaries.adversarial_learning import AdversarialLearning
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adlib.learners.simple_learner import SimpleLearner
from sklearn.linear_model import LinearRegression

dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400', binary=False, raw=True)
training_, testing_ = dataset.split({'train': 70, 'test': 30})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)

# set learner and basic attributes
learner_model = LinearRegression()
basic_learner = SimpleLearner(model=learner_model, training_instances=training_data)
basic_learner.train()

attacker = AdversarialLearning(threshold=10, learner=basic_learner)
attacker.set_adversarial_params(learner=basic_learner, training_instances=training_data)
attacked_instances = attacker.attack(testing_data)

predictions1 = basic_learner.predict(testing_data)
predictions2 = basic_learner.predict(attacked_instances)

print(predictions1, predictions2)
