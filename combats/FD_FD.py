import sys
sys.path.append("/home/dingx/adlib/adlib")
from sklearn import svm
from learners import FeatureDeletion
from timeit import default_timer as timer
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.feature_deletion import AdversaryFeatureDeletion
from sklearn import metrics
from utils import save_result
import multiprocessing
import pandas as pd
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
    train_,test_ = _dataset.split(0.7)
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
    attacker = AdversaryFeatureDeletion()
    attacker.set_params(par_map)
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
    return ret


def generate_param_map(param_path = "FD_CG_para", process_time=10):
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
    param_path = sys.argv[1]
    return_path = sys.argv[2]
    lst = generate_param_map(param_path,process_time=5)
    pool = multiprocessing.Pool(processes=5)
    result = pool.map(run, lst)
    arr = np.array(result)
    data = pd.DataFrame(arr, columns = ["old_acc", "old_prec", "old_rec", "old_f1","learn_t",
                "new_acc", "new_prec","new_rec", "new_f1","atk_t"])
    data.to_csv(return_path, sep='\t', encoding='utf-8')