
from sklearn import svm
from learners import SimpleLearner
import adversaries as adversaries
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance
from data_reader.operations import load_dataset
from adversaries.restrained_attack import Restrained
from adversaries.coordinate_greedy import CoordinateGreedy

dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/full',binary= False,raw=True)
training_, testing_ = dataset.split({'train': 60, 'test': 40})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)


#test simple learner svm
learning_model = svm.SVC(probability=True, kernel='linear')
learner1 = SimpleLearner(learning_model, training_data)
learner1.train()
predictions = learner1.predict(testing_data)
print(predictions)
#print([testing_data[i].label for i in range(len(testing_data))])


#test Restrained_attack
attacker = Restrained(binary = False, learner = learner1)
attacker.set_adversarial_params(learner1, training_data)
new_testing_data = attacker.attack(testing_data)


predictions2 = learner1.predict(new_testing_data)
print(predictions2)

#test retraining
learner2 = learner.Retraining(learning_model,training_data,attacker=attacker)
attacker.set_adversarial_params(learner2, training_data)
learner2.train()
predictions3 = learner2.predict(testing_data)
print(predictions3)
val = [testing_data[i].label for i in range(len(testing_data))]

#test coordinate_greedy
attacker2 = CoordinateGreedy(learner=learner1, max_change = 500)
new_data2 = attacker2.attack(testing_data)

predictions4 = learner1.predict(new_data2)
print(predictions4)

print("\n")
print(val)

#compare result
sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0
for i  in range (len(val)):
    if val[i] == 1 and predictions3[i] == -1:
        sum1 += 1
    if val[i] == 1 and predictions2[i] == -1:
        sum2 += 1
    if val[i] == 1 and  predictions[i] == -1:
        sum3 += 1
    if val[i] == 1 and predictions4[i] == -1:
        sum4 += 1

print(sum1,sum2,sum3,sum4)




