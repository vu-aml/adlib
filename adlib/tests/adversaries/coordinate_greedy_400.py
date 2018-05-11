from sklearn import svm
from adlib.learners import SimpleLearner
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.coordinate_greedy import CoordinateGreedy
from sklearn import metrics

# Failing

def summary(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("lengths of two label lists do not match")
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return "accuracy: {0} \n precision: {1} \n recall: {2}\n".format(acc, prec, rec)


def get_evasion_set(x_test, y_pred):
    # for x, y in zip(x_test, y_pred):
    #     print("true label: {0}, predicted label: {1}".format(x.label, y))
    ls = [x for x, y in zip(x_test, y_pred) if x.label==1 and y==1]
    print("{0} malicious instances are being detected initially".format(len(ls)))
    return ls, [x.label for x in ls]


dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',binary= False,raw=True)
training_, testing_ = dataset.split({'train': 60, 'test': 40})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)
test_true_label = [x.label for x in testing_data]


#test simple learner svm
learning_model = svm.SVC(probability=True, kernel='linear')
learner1 = SimpleLearner(learning_model, training_data)
learner1.train()

predictions = learner1.predict(testing_data)
print("======== initial prediction =========")
print(summary(predictions, test_true_label))


#test Restrained_attack
attacker = CoordinateGreedy(lambda_val=0, max_change=100, step_size=1000)
attacker.set_adversarial_params(learner1, testing_data)
new_testing_data = attacker.attack(testing_data)


predictions2 = learner1.predict(new_testing_data)
print("========= post-attack prediction =========")
print(summary(predictions2, test_true_label))

