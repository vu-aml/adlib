from sklearn import svm
from learners import SimpleLearner
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.gradient_descent import GradientDescent
from sklearn import metrics

def summary(y_pred, y_true):
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
    s = "accuracy: {0} \n precision: {1} \n recall: {2}\n".format(acc, prec, rec)
    s += "TP {0}, FP {1}, TN {2}, FN {3}".format(tp,fp,tn,fn)
    return s

_dataset = EmailDataset(path='../data_reader/data/uci/uci_modified.csv', binary=False, raw=False)
train_, test_ = _dataset.split(0.8)
training_data = load_dataset(train_)
testing_data = load_dataset(test_)
test_true_label = [x.label for x in testing_data]


#test simple learner svm
learning_model = svm.SVC(probability=True, kernel='linear')
learner1 = SimpleLearner(learning_model, training_data)
learner1.train()

predictions = learner1.predict(testing_data)
print("======== initial prediction =========")
print(summary(predictions, test_true_label))


adversary = GradientDescent(step_size=2,trade_off=10,max_iter=1000,mimicry="kde_euclidean")
adversary.set_adversarial_params(learner1,training_data)
new_testing_data = adversary.attack(testing_data)

predictions2 = learner1.predict(new_testing_data)
print("========= post-attack prediction =========")
print(summary(predictions2, test_true_label))