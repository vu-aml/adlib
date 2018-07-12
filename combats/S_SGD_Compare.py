import sys
sys.path.append("/home/dingx/adlib")
from timeit import default_timer as timer
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.simple_gradient_descent import SimpleGradientDescent
from learners.simple_learner import SimpleLearner
from learners.l_infinity_svm import L_infSVM
from sklearn import svm
from sklearn import metrics
import multiprocessing
import pandas as pd
import numpy as np
import json
import logging

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
    logging.info(s)
    print(s)
    acc = np.around(metrics.accuracy_score(y_true, y_pred), 3)
    prec = np.around(metrics.precision_score(y_true, y_pred), 3)
    rec = np.around(metrics.recall_score(y_true, y_pred), 3)
    f1 = np.around(metrics.f1_score(y_true, y_pred), 3)
    result = [acc, prec, rec, f1]
    print(result)
    return result


def run(par_map):
    _dataset = EmailDataset(path='../data_reader/data/trec07p/delay', binary=False, raw=True,max_features_=200, norm="l2")
    train_, test_ = _dataset.split(0.8)
    training_data = load_dataset(train_)
    testing_data = load_dataset(test_)
    test_true_label = [x.label for x in testing_data]

    # running feature deletion learner
    # the parameter should be altered by process arguments
    l_start = timer()
    learning_model = svm.SVC(probability=True, kernel='linear', C=0.5)
    simple_learner = SimpleLearner(model=learning_model,training_instances=training_data)
    simple_learner.set_params(params=par_map)
    simple_learner.train()
    l_end = timer()

    #running Linfinity
    lin_learner = L_infSVM(training_instances=training_data)
    lin_learner.set_params(params = par_map)
    lin_learner.train()



    predictions = simple_learner.predict(testing_data)
    print("The svm classification before attack finishes.")
    result1 = single_run_list(predictions, test_true_label)
    result1.append(np.around((l_end - l_start), 3))

    predictions2 = lin_learner.predict(testing_data)
    print("The linfity classification before attack finishes.")
    result1 = single_run_list(predictions2, test_true_label)
    result1.append(np.around((l_end - l_start), 3))

    # test Restrained_attack
    a_start = timer()
    attacker = SimpleGradientDescent()
    attacker.set_params(par_map)
    attacker.set_adversarial_params(simple_learner, testing_data)
    attacked_data = attacker.attack(testing_data)

    attacked_predictions = simple_learner.predict(attacked_data)
    attacked_predictions_2 = lin_learner.predict(attacked_data)

    a_end = timer()
    print("The svm classification after attack finishes.")
    result2 = single_run_list(attacked_predictions, test_true_label)
    result2.append(np.around((a_end - a_start), 3))

    print("The Linfinity classification after attack finishes.")
    result2 = single_run_list(attacked_predictions_2, test_true_label)
    result2.append(np.around((a_end - a_start), 3))

    ret = []
    ret.extend(result1)
    ret.extend(result2)
    return ret


def generate_param_map(par_map, process_time=10):
    lst = []
    for item in (par_map["param"]):
        for i in range(process_time):
            lst.append(item)
    return lst


def generate_interval(start, end, process_count, dtype=None, log=False):
    if log:
        return (np.logspace(start, end, process_count)).tolist()
    return (np.linspace(start, end, process_count, dtype=dtype)).tolist()


def generate_index(param_lst):
    map = []
    for item in param_lst:
        title = "Simple_SVM " + "C" + str(item["C"]) + " SGD" + " S" + str \
                    (item["step_size"]) + " MI" + str(item["max_iter"]) + " bound" + str(item["bound"])
        map.append(title)
    return map


if __name__ == '__main__':
    logging.basicConfig(filename='SSGD_result.log', level=logging.INFO)
    param_path = sys.argv[1]
    data_path = sys.argv[2]
    process_time = int(sys.argv[3])
    #exl_path = sys.argv[4]
    with open(param_path, 'r') as para_file:
        par_map = json.load(para_file)
    total_time = len(par_map["param"]) * process_time

    lst = generate_param_map(par_map, process_time)
    pool = multiprocessing.Pool(processes=total_time)

    result = pool.map(run, lst)
    #arr = np.array(result)
    #title_map = generate_index(lst)
    #data = pd.DataFrame(arr, columns=["old_acc", "old_prec", "old_rec", "old_f1", "learn_t",
    #                                  "new_acc", "new_prec", "new_rec", "new_f1", "atk_t"],index = list(title_map))
    #data.to_csv(data_path, sep='\t', encoding='utf-8')
    #data.to_excel(data_path)
