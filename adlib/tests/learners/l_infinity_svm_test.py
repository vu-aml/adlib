from sklearn import svm
from learners import SimpleLearner
from learners import L_infSVM
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from learners.feature_deletion import FeatureDeletion
from adversaries.feature_deletion import AdversaryFeatureDeletion
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt

# passed

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

dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',binary= False,raw=True)
training_, testing_ = dataset.split({'train': 86, 'test': 14})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)
test_true_label = [x.label for x in testing_data]

print("training data shape:" + str(dataset.shape))
print("num of training data:" + str(len(training_data)))
#test simple learner svm
learner1 = L_infSVM(training_data, 0)
learner1.train()

predictions = learner1.predict(testing_data)
print(summary(predictions, test_true_label))
