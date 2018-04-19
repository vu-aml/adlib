import sys
sys.path.append("/home/jason/Jason/adlib")
from sklearn import svm
from learners import Retraining
from timeit import default_timer as timer

from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.coordinate_greedy import CoordinateGreedy
from sklearn import metrics
import multiprocessing
import numpy as np


def single_run_list(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("lengths of two label lists do not match")
    tp, fp, tn, fn = 0,0,0,0
    for idx in range(len(y_true)):
        if y_true[idx] == 1 and y_true[idx]==y_pred[idx]:
            tp += 1
        if y_true[idx] == 1 and y_true[idx] != y_pred[idx]:
            fn += 1
        if y_true[idx] == -1 and y_true[idx] == y_pred[idx]:
            tn += 1
        if y_true[idx] == -1 and y_true[idx] != y_pred[idx]:
            fp += 1
    acc = np.around(metrics.accuracy_score(y_true, y_pred), 3)
    prec = np.around(metrics.precision_score(y_true, y_pred), 3)
    rec = np.around(metrics.recall_score(y_true, y_pred), 3)
    f1 = np.around(metrics.f1_score(y_true, y_pred), 3)

    # result = dict(accuracy=acc, precision=prec, recall=rec)
    result = [acc, prec, rec, f1]
    return result


dataset = EmailDataset(path='./data_reader/data/enron/index_dir_small', binary=False, raw=True, max_features_=500)
training_, testing_ = dataset.split(0.01)
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)
test_true_label = [x.label for x in testing_data]

# test simple learner svm
l_start = timer()
base_model = svm.SVC(kernel="linear")
learner1 = Retraining(base_model, training_data, CoordinateGreedy)
learner1.set_params({'adv_params': {'max_change': 80, 'lambda_val': 0.05}, 'iterations': 3})
learner1.train()
l_end = timer()

predictions = learner1.predict(testing_data)
# print("======== initial prediction =========")
# print(summary(predictions, test_true_label))
result1 = single_run_list(predictions, test_true_label)
result1.append(np.around((l_end - l_start), 3))

# test attack
a_start = timer()
attacker = CoordinateGreedy(max_change=800, lambda_val=0)
attacker.set_adversarial_params(learner1, testing_data)
new_testing_data = attacker.attack(testing_data)

predictions2 = learner1.predict(new_testing_data)
result2 = single_run_list(predictions2, test_true_label)
a_end = timer()
result2.append(np.around((a_end - a_start), 3))
ret = []
ret.extend(result1)
ret.extend(result2)

print(ret)
