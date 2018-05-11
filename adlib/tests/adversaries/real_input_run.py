from sklearn import svm
from adlib.learners import SimpleLearner
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.restrained_attack import Restrained
from adversaries.coordinate_greedy import CoordinateGreedy
from sklearn import metrics


def summary(y_pred, y_true):
    # if len(y_pred) != len(y_true):
    #     raise ValueError("lengths of two label lists do not match")
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return "accuracy: {0} \n precision: {1} \n recall: {2}\n".format(acc, prec, rec)


def get_evasion_set(x_test, y_pred):
    for x, y in zip(x_test, y_pred):
        print("true label: {0}, predicted label: {1}".format(x.label, y))
    ls = [x for x, y in zip(x_test, y_pred) if x.label==1 and y==1]
    print("{0} malicious instances are being detected initially")
    return ls, [x.label for x in ls]


dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',binary= False,raw=True)
training_, testing_ = dataset.split({'train': 60, 'test': 40})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)
test_true_label = [x.label for x in testing_data]


#test simple learner svm
learning_model = svm.SVC(probability=True, kernel='linear')
learner1 = SimpleLearner(learning_model, training_data)
learner1.train()

predictions = learner1.predict(testing_data)
print("======== initial prediction =========")
print(summary(predictions, test_true_label))

print("finding the set of detected malicious instances for attacking")
mal_set, mal_label = get_evasion_set(testing_data, predictions)
print("size of new dataset: {0}".format(len(mal_set)))

#test Restrained_attack
attacker = Restrained(f_attack=0.99, binary = False, learner = learner1)
attacker.set_adversarial_params(learner1, training_data)
new_testing_data = attacker.attack(mal_set)


predictions2 = learner1.predict(new_testing_data)
print("========= post-attack prediction =========")
print(summary(predictions2, mal_label))

# #test retraining
# learner2 = learner.Retraining(learning_model,training_data,attacker=attacker)
# attacker.set_adversarial_params(learner2, training_data)
# learner2.train()
# predictions3 = learner2.predict(testing_data)
# print(predictions3)
# val = [testing_data[i].label for i in range(len(testing_data))]
#
# #test coordinate_greedy
# attacker2 = CoordinateGreedy(learner=learner1, max_change = 500)
# new_data2 = attacker2.attack(testing_data)
#
# predictions4 = learner1.predict(new_data2)
# print(predictions4)
#
# print("\n")
# print(val)
#
# #compare result
# sum1 = 0
# sum2 = 0
# sum3 = 0
# sum4 = 0
# for i  in range (len(val)):
#     if val[i] == 1 and predictions3[i] == -1:
#         sum1 += 1
#     if val[i] == 1 and predictions2[i] == -1:
#         sum2 += 1
#     if val[i] == 1 and  predictions[i] == -1:
#         sum3 += 1
#     if val[i] == 1 and predictions4[i] == -1:
#         sum4 += 1
#
# print(sum1,sum2,sum3,sum4)


