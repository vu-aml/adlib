import sys
from sklearn import svm
from data_reader import binary_input
from sklearn import metrics
import learners as learner
import adversaries as ad
import random
from data_reader.operations import sparsify
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset

def main(argv):
    """
    driver class that performs demo of the library
    """

    # pre-process data and randomly partition
    dataset = EmailDataset(path='./data_reader/data/test/100_instance_debug.csv', raw=False)
    training_, testing_ = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)


    # initialize sklearn model
    learning_model = svm.SVC(probability=True, kernel='linear')

    # initialize and train RobustLearner
    attacker = ad.BinaryGreedy()
    clf2 = learner.Retraining(learning_model,training_data,attacker=attacker)
    attacker.set_adversarial_params(clf2,training_data)
    clf2.train()

    # produce simple metrics
    y_predict = clf2.predict(testing_data[0])
    y,X = sparsify(testing_data)
    y_true = y[0]
    print(y_predict,y_true)
    score = metrics.precision_score([y_predict],[y_true])
    print("score = "+str(score))



if __name__ == "__main__":
  main(sys.argv[1:])
