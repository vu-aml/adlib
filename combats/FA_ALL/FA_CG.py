import sys
sys.path.append("/home/dingx/adlib")
from learners import FeatureDeletion
from timeit import default_timer as timer
from data_reader.save_load import load
from adversaries import FreeRange
from adversaries import CoordinateGreedy
from sklearn import metrics
from utils import save_result
import multiprocessing
import pandas as pd
import numpy as np
import json
import logging
import openpyxl
import copy
import pickle


def _pickle(outfile, save=True, weight= None, bias = None, time = None):
    """A fast method for saving and loading weight and bias (ndarray objects)
    Args:
        outfile (str): The destination file.
        save (boolean, optional): If True, serialize, if False, load.

    """
    if save:
        with open(outfile, 'wb+') as fileobj:
            pickle.dump({
                'bias': bias,
                'weight': weight,
                'time':time
            }, fileobj, pickle.HIGHEST_PROTOCOL)
    else:
        with open(outfile, 'rb') as fileobj:
            data = pickle.load(fileobj)
            return data

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
    s = "TP {0}, FP {1}, TN {2}, FN {3}".format(tp, fp, tn, fn)
    print(s)
    acc = np.around(metrics.accuracy_score(y_true, y_pred), 3)
    prec = np.around(metrics.precision_score(y_true, y_pred), 3)
    rec = np.around(metrics.recall_score(y_true, y_pred), 3)
    f1 = np.around(metrics.f1_score(y_true, y_pred), 3)
    result = [acc, prec, rec, f1]
    return result


def run(par_map):
    print("starts")
    training_data = load(("../../data_reader/data/transformed/training_"+ str(par_map['file']) +".pkl"),binary=False)

    testing_data = load(("../../data_reader/data/transformed/testing_" + str(par_map['file']) +".pkl"), binary=False)
    test_true_label = [x.label for x in testing_data]

    # running feature deletion learner
    # the parameter should be altered by process arguments
    path = "FD_R/FD" + "H" + str(par_map["hinge_loss_multiplier"]) + "F" + \
           str(par_map['max_feature_deletion']) + str(par_map["file"]) + ".pkl"
    data = _pickle(outfile=path,save=False)
    time = data["time"]
    weight = data["weight"]
    bias = data["bias"]



    fd_learner = FeatureDeletion(weight_vector=weight,bias=bias,training_instances=training_data,pre_trained=True)
    predictions = fd_learner.predict(testing_data)
    print("The classification before attack finishes.")
    result1 = single_run_list(predictions, test_true_label)
    result1.append(np.around(time, 3))

    # test Restrained_attack
    a_start = timer()
    attacker = CoordinateGreedy()
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



def generate_param_map(par_map, process_time=10):
    lst = []
    for item in (par_map["param"]):
        for i in range(process_time):
            lst.append(copy.copy(item))
            lst[-1]['file'] = i+11
    return lst


def generate_interval(start, end, process_count, dtype=None, log=False):
    if log:
        return (np.logspace(start, end, process_count)).tolist()
    return (np.linspace(start, end, process_count, dtype=dtype)).tolist()


def generate_index(param_lst):
    map = []
    for item in param_lst:
        title = "FD " + "H" + str(item["hinge_loss_multiplier"]) + " F" + str(
            item["max_feature_deletion"]) + " CG" + " Iter" + str(item["max_change"]) +\
                " L" + str(item["lambda_val"]) + " Step" + str(item['step_size'])
        map.append(title)
    return map



if __name__ == '__main__':
    #logging.basicConfig(filename='FDFD_result.log', level=logging.INFO)
    param_path = sys.argv[1]
    data_path = sys.argv[2]
    with open(param_path, 'r') as para_file:
        par_map = json.load(para_file)
    total_time = len(par_map["param"]) * 5

    lst = generate_param_map(par_map, 5)
    pool = multiprocessing.Pool(processes=total_time)

    result = pool.map(run, lst)
    arr = np.array(result)
    title_map = generate_index(lst)
    data = pd.DataFrame(arr, columns=["old_acc", "old_prec", "old_rec", "old_f1", "learn_t",
                                      "new_acc", "new_prec", "new_rec", "new_f1", "atk_t"],index = list(title_map))
    data.to_excel(data_path)