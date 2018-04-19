import sys
sys.path.append("/home/jason/Jason/adlib")
from sklearn import svm
from learners import SVMRestrained
from timeit import default_timer as timer
import learners as learner
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

def run(custom_var=1):
    dataset = EmailDataset(path='./data_reader/data/enron/index_dir_small', binary=False, raw=True, max_features_=500)
    training_, testing_ = dataset.split(0.01)
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)
    test_true_label = [x.label for x in testing_data]

    #test FREERAGE learner svm
    l_start = timer()
    # learning_model = svm.SVC(probability=True, kernel='linear')
    # learner1 = learner.SimpleLearner(learning_model, training_data)
    learner1 = SVMRestrained(training_instances=training_data)
    learner1.set_params({"c": 1, "c_delta": custom_var})
    learner1.train()

    l_end = timer()
    predictions = learner1.predict(testing_data)
    # print("======== initial prediction =========")
    # print(summary(predictions, test_true_label))
    result1 = single_run_list(predictions, test_true_label)
    result1.append(np.around((l_end - l_start), 3))

    # attack
    a_start = timer()
    attacker = CoordinateGreedy(max_change=800, lambda_val=0.001)
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
    pool = multiprocessing.Pool(processes=2)
    varyingParam = generate_interval(0.85, 0.85, 2, log=False)

    result_list = pool.map(run, varyingParam)
    with open('LINF_FD.txt', 'a') as f:
        s = "param,  old_acc,  old_prec,  old_rec,  old_f1, learn_t"+\
                ",  new_acc,  new_prec,  new_rec,  new_f1,   atk_t"
        f.write(s+"\n")
        print(s)
        for r in result_list:
            f.write(r+"\n")
            print(r)

    result_mat = [r.split(",  ") for r in result_list]
    # print(result_mat)
    result_mat = [[float(i) for i in r] for r in result_mat]
    # print(result_mat)
    result_avg = list(map(list, zip(*result_mat)))
    # for l in result_avg:
    #     l.sort(reverse=True)
    # result_avg = [l[:10] for l in result_avg]

    result_avg = [sum(l)/len(l) for l in result_avg]
    result_avg = [round(float(i), 3) for i in result_avg]
    print(result_avg)
    print("old accuracy: {}; new accuracy: {}; accuracy change: {}"
          .format(result_avg[1], result_avg[6], round(float(result_avg[6]-result_avg[1]), 3)))
    print("old f1: {}; new f1: {}; f1 change: {}"
          .format(result_avg[4], result_avg[9], round(float(result_avg[9]-result_avg[4]), 3)))
    print("old recall: {}; new recall: {}; recall change: {}"
          .format(result_avg[3], result_avg[8], round(float(result_avg[8]-result_avg[3]), 3)))
    print("training time: {}; attacking time {}".format(result_avg[5], result_avg[10]))
