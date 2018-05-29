import sys
from sklearn import svm
from learners import FeatureDeletion
from timeit import default_timer as timer
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.coordinate_greedy import CoordinateGreedy
from sklearn import metrics
from utils import save_result
import multiprocessing
import numpy as np
import json

def single_run_list(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("lengths of two label lists do not match")
    tp, fp, tn, fn = 0, 0, 0, 0
    for idx in range(len(y_true)):
        if y_true[idx] == 1 and y_true[idx] == y_pred[idx]:
            tp += 1
        if y_true[idx] == 1 and y_true[idx] != y_pred[idx]:
            fn += 1
        if y_true[idx] == -1 and y_true[idx] == y_pred[idx]:
            tn += 1
        if y_true[idx] == -1 and y_true[idx] != y_pred[idx]:
            fp += 1
    print (tp,fp,tn,fn)
    acc = np.around(metrics.accuracy_score(y_true, y_pred), 3)
    prec = np.around(metrics.precision_score(y_true, y_pred), 3)
    rec = np.around(metrics.recall_score(y_true, y_pred), 3)
    f1 = np.around(metrics.f1_score(y_true, y_pred), 3)
    result = [acc, prec, rec, f1]
    return result

def run(par_map):
    _dataset = EmailDataset(path='../data_reader/data/enron/10', binary=False, raw=True, max_features_=500)
    train_,test_ = _dataset.split(0.8)
    training_data = load_dataset(train_)
    testing_data = load_dataset(test_)
    test_true_label = [x.label for x in testing_data]

    #running feature deletion learner
    #the parameter should be altered by process arguments
    l_start = timer()
    fd_learner = FeatureDeletion(training_data, params=par_map)
    fd_learner.train()
    l_end = timer()

    predictions = fd_learner.predict(testing_data)
    print("The classification before attack finishes.")
    result1 = single_run_list(predictions, test_true_label)
    result1.append(np.around((l_end - l_start), 3))

    #test Restrained_attack
    a_start = timer()
    attacker = CoordinateGreedy(max_change=100, lambda_val=0.1)
    attacker.set_adversarial_params(fd_learner, testing_data)
    attacked_data = attacker.attack(testing_data)

    attacked_predictions = fd_learner.predict(attacked_data)

    result2 = single_run_list(attacked_predictions, test_true_label)
    a_end = timer()
    print("The classification after attack finishes.")
    result2.append(np.around((a_end - a_start), 3))
    ret = []
    ret.extend(result1)
    ret.extend(result2)
    name = "FD:C=3,K=20;CG:M=100,L=0.1"
    return name,ret


def generate_param_map(param_path = "para", process_time=10):
    with open(param_path, 'r') as para_file:
        par_map= json.load(para_file)
    lst = []
    for i in range(process_time):
        lst.append(par_map)
    return lst

def generate_interval(start, end, process_count, dtype=None, log=False):
    if log:
        return (np.logspace(start, end, process_count)).tolist()
    return (np.linspace(start, end, process_count, dtype=dtype)).tolist()


if __name__ == '__main__':
    data_path = sys.argv(0)
    param_path = sys.argv(1)
    lst = generate_param_map(param_path)
    pool = multiprocessing.Pool(processes=10)
    result = pool.map(run, pool)