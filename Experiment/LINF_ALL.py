import sys
sys.path.append("/home/zhanj16/adlib")
from sklearn import svm
from learners import L_infSVM
from timeit import default_timer as timer
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.good_word import GoodWord
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

def run(custom_var=10):
    dataset = EmailDataset(path='./data_reader/data/enron/index_dir_small', binary=False, raw=True)
    training_, testing_ = dataset.split(0.3)
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)
    test_true_label = [x.label for x in testing_data]

    #test simple learner svm
    l_start = timer()
    learner1 = L_infSVM(training_data, 7)
    learner1.train()
    l_end = timer()

    predictions = learner1.predict(testing_data)
    # print("======== initial prediction =========")
    # print(summary(predictions, test_true_label))
    result1 = single_run_list(predictions, test_true_label)
    result1.append(np.around((l_end - l_start), 3))

    #test Restrained_attack
    a_start = timer()
    attacker = GoodWord(n=custom_var)
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


def generate_interval(start, end, process_count, dtype=None, log=False):
    if log:
        return (np.logspace(start, end, process_count)).tolist()
    return (np.linspace(start, end, process_count, dtype=dtype)).tolist()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=30)
    varyingParam = generate_interval(0, 150, 30, dtype=int)

    result_list = pool.map(run, varyingParam)
    with open('LINF_FD.txt', 'a') as f:
        s = "param,  old_acc,  old_prec,  old_rec,  old_f1, learn_t"+\
                ",  new_acc,  new_prec,  new_rec,  new_f1,   atk_t"
        f.write(s+"\n")
        print(s)
        for r in result_list:
            f.write(r+"\n")
            print(r)
