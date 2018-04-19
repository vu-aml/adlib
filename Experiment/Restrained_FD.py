import sys
sys.path.append("/home/zhanj16/adlib")
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
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    # result = dict(accuracy=acc, precision=prec, recall=rec)
    result = [acc, prec, rec]
    return result

def run(var):
    dataset = EmailDataset(path='./data_reader/data/enron/enron1/index_dir', binary=False, raw=True)
    training_, testing_ = dataset.split(0.3, 77)
    training_data = load_dataset(training_)
    print("number of training data: {}".format(len(training_data)))

    testing_data = load_dataset(testing_)
    test_true_label = [x.label for x in testing_data]

    #test simple learner svm
    l_start = timer()
    # learning_model = svm.SVC(probability=True, kernel='linear')
    # learner1 = learner.SimpleLearner(learning_model, training_data)
    learner1 = SVMFreeRange(training_instances=training_data)
    learner1.set_params({"c": var})
# try:
    learner1.train()
# except Exception as e:
#     return ',  '.join(map(str, [0]*9))
    l_end = timer()

    predictions = learner1.predict(testing_data)
    # print("======== initial prediction =========")
    # print(summary(predictions, test_true_label))
    result1 = single_run_list(predictions, test_true_label)
    result1.append(l_end -l_start)

    #test Restrained_attack
    a_start = timer()
    attacker = AdversaryFeatureDeletion(learner1, 30)
    attacker.set_adversarial_params(learner1, testing_data)
    new_testing_data = attacker.attack(testing_data)

    predictions2 = learner1.predict(new_testing_data)
    result2 = single_run_list(predictions2, test_true_label)
    a_end = timer()
    result2.append(a_end - a_start)
    ret = [var]
    ret.extend(result1)
    ret.extend(result2)
    return ',  '.join(map(str, ret))


def generate_interval(start, end, process_count, dtype=None, log=False):
    if log:
        return (np.logspace(start, end, process_count)).tolist()
    return (np.linspace(start, end, process_count, dtype=dtype)).tolist()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=1)
    varyingParam = generate_interval(0.5, 0.5, 1)

    result_list = pool.map(run, varyingParam)
    with open('LINF_FD_fdcount.txt', 'a') as f:
        s = "FR_cf,  old_accuracy,  old_precision,  old_recall,  learning_time"+\
                ",  new_accuracy,  new_precision,  new_recall,  attack_time"
        f.write(s+"\n")
        print(s)
        for r in result_list:
            f.write(r+"\n")
            print(r)
