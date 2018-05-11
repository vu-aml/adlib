from adversaries.cost_sensitive import CostSensitive
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adlib.learners.adversary_aware import AdversaryAware
from adlib.learners.simple_learner import SimpleLearner
from sklearn.naive_bayes import GaussianNB


 #TODO:need a detailed look at the algorithm and more tests

dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',binary= False,raw=True)
training_, testing_ = dataset.split({'train': 60, 'test': 40})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)

#set learner and basic attributes
learner_model = GaussianNB()
basic_learner = SimpleLearner(model=learner_model, training_instances=training_data)
basic_learner.train()
attacker = CostSensitive(binary=False)
param = {}
param['Ua'] = [[0, 20], [0, 0]]
param['Vi'] = 0
param['Uc'] = [[1, -1], [-10, 1]]
param['scenario'] = None
attacker.set_params(param)
attacker.set_adversarial_params(learner=basic_learner,training_instances=training_data)


learner = AdversaryAware(attacker=attacker,training_instances=training_data)
learner.train()

print(learner.get_params())
print(learner.predict(testing_data))




