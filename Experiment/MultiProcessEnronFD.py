import sys
sys.path.append("/home/zhanj16/abc/adlib")
from sklearn import svm
from learners import SimpleLearner
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.feature_deletion import AdversaryFeatureDeletion
from sklearn import metrics
import multiprocessing
import numpy as np

def single_run_dict(y_pred, y_true):
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
    result = dict(accuracy=acc, precision=prec, recall=rec)
    return result

def run(del_cnt=10):
    dataset = EmailDataset(path='../data_reader/data/enron/enron1/index_dir',binary= False,raw=True)
    training_, testing_ = dataset.split(0.6)
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)
    test_true_label = [x.label for x in testing_data]

    #test simple learner svm
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner1 = SimpleLearner(learning_model, training_data)
    learner1.train()

    predictions = learner1.predict(testing_data)
    # print("======== initial prediction =========")
    # print(summary(predictions, test_true_label))
    result1 = single_run_dict(predictions, test_true_label)


    #test Restrained_attack
    attacker = AdversaryFeatureDeletion(num_deletion=del_cnt)
    attacker.set_adversarial_params(learner1, testing_data)
    new_testing_data = attacker.attack(testing_data)

    predictions2 = learner1.predict(new_testing_data)
    result2 = single_run_dict(predictions2, test_true_label)

    s = "run with variable param value: {0}\n".format(del_cnt)
    s += str(result1) + "\n"
    s += str(result2) + "\n"
    return s


def generate_interval(start, end, process_count, dtype=None):
    return (np.linspace(start, end, process_count, dtype=dtype)).tolist()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    varyingParam = generate_interval(1,100, 4, dtype=int)

    result_list = pool.map(run, varyingParam)
    with open('FD_result.txt', 'a') as the_file:
        for r in result_list:
            the_file.write(r)
            print(r)
