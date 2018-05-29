import sys
from sklearn import svm
from learners import SVMFreeRange
from timeit import default_timer as timer
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.feature_deletion import AdversaryFeatureDeletion
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
    print(tp,fp,tn,fn)

    # result = dict(accuracy=acc, precision=prec, recall=rec)
    result = [acc, prec, rec, f1]
    return result

def run(custom_var=1):
    #training_data: 600*500 feature
    training_dataset = EmailDataset(path='./data_reader/data/enron/1', binary=False, raw=True, max_features_=500)
    testing_dataset = EmailDataset(path= './data_reader/data/enron/2',binary= False, raw= True, max_features_=500)
    training_data = load_dataset(training_dataset)
    testing_data = load_dataset(testing_dataset)
    test_true_label = [x.label for x in testing_data]

    #test FREERAGE learner svm
    l_start = timer()
    # learning_model = svm.SVC(probability=True, kernel='linear')
    # learner1 = learner.SimpleLearner(learning_model, training_data)
    learner1 = SVMFreeRange(training_instances=training_data)
    learner1.set_params({"c": 1, "c_f": custom_var})
    learner1.train()

    l_end = timer()
    predictions = learner1.predict(testing_data)
    # print("======== initial prediction =========")
    # print(summary(predictions, test_true_label))
    result1 = single_run_list(predictions, test_true_label)
    result1.append(np.around((l_end - l_start), 3))

    # test attack
    a_start = timer()
    attacker = CoordinateGreedy(max_change=80, lambda_val=0.01)
    attacker.set_adversarial_params(learner1, testing_data)
    new_testing_data = attacker.attack(testing_data)

    predictions2 = learner1.predict(new_testing_data)
    result2 = single_run_list(predictions2, test_true_label)
    a_end = timer()
    result2.append(np.around((a_end - a_start), 3))
    ret = [np.around(custom_var, 3)]
    ret.extend(result1)
    ret.extend(result2)
    return ',  '.join(map(str, ret))