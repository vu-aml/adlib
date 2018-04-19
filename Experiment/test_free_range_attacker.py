from sklearn import svm
from learners import SimpleLearner
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.free_range_attack import FreeRange
from sklearn import metrics
from learners import svm_freerange
from learners import svm_restrained
import os


def summary(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("lengths of two label lists do not match")
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return "accuracy: {0} \n precision: {1} \n recall: {2}\n".format(acc, prec, rec)


def get_evasion_set(x_test, y_pred):
    # for x, y in zip(x_test, y_pred):
    #     print("true label: {0}, predicted label: {1}".format(x.label, y))
    ls = [x for x, y in zip(x_test, y_pred) if x.label==1 and y==1]
    print("{0} malicious instances are being detected initially")
    return ls, [x.label for x in ls]

dataset = EmailDataset(path='./data_reader/data/enron/enron1/index_dir',binary= False,raw=True)
#dataset2= EmailDataset(path='./data_reader/data/uci/spambase.csv', binary = False, raw = False)
#training2_,testing2_ = dataset.split(0.6)
training_, testing_ = dataset.split(0.1)
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)
#training2_data = load_dataset(training2_)
#testing2_data = load_dataset(testing2_)
test_true_label = [x.label for x in testing_data]


#test simple learner svm
learning_model = svm.SVC(probability=True, kernel='linear')
learner1 = SimpleLearner(learning_model, training_data)
learner1.train()
print("Simple learner done.")

#test svm restrained
#learner3 = svm_restrained.SVMRestrained(params=None,training_instances=training2_data)
#learner3.train()
#print("tranining svm restrained")
#predictions3 = learner3.predict(testing2_data)

#test free range svm
learner2 = svm_freerange.SVMFreeRange(params= None,training_instances=training_data)
print("training_svm_freerange")
learner2.train()
print("SVM done.")



predictions = learner1.predict(testing_data)
predictions2 = learner2.predict(testing_data)
print("======== initial prediction for Simple Learner =========")
print(summary(predictions, test_true_label))
print("======== initial prediction for SVM Freereange =========")
print(summary(predictions2, test_true_label))

print("finding the set of detected malicious instances for attacking")
mal_set, mal_label = get_evasion_set(testing_data, predictions)
print("size of new dataset: {0}".format(len(mal_set)))

#test Restrained_attack
attacker = FreeRange(0.1)
attacker.set_adversarial_params(learner1, testing_data)
new_testing_data = attacker.attack(mal_set)


predictions_post = learner1.predict(new_testing_data)
predictions_post2 = learner2.predict(new_testing_data)
print("========= post-attack prediction simple learner=========")
print(summary(predictions_post, mal_label))
print("======== post prediction for SVM Freereange =========")
print(summary(predictions_post2, mal_label))
