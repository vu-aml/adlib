import sys
from sklearn import svm
from data_reader.binary_input import Instance
from data_reader.operations import load_dataset
from sklearn import metrics
import learners as learner
import random
from random import seed, shuffle
from data_reader.dataset import EmailDataset
import matplotlib.pyplot as plt


def main(argv):
    """
    driver class that performs demo of the library
    """

    # pre-process data and randomly partition
    dataset = EmailDataset(path='../../data_reader/data/test/100_instance_debug.csv', raw=False)
    training_, testing_ = dataset.split({'train': 60, 'test': 40})
    training_data = load_dataset(training_)
    testing_data = load_dataset(testing_)

    # initialize and train RobustLearner
    clf2 = learner.FeatureDeletion(training_data,{'hinge_loss_multiplier': 1,
                                    'max_feature_deletion': 0})
    clf2.train()

    # produce simple metrics
    y_predict = clf2.predict(testing_data[0])
    y_true = testing_data[0].label
    print(y_predict,y_true)

    score = metrics.accuracy_score([y_true], [y_predict])
    print("score = "+str(score))

    wgt = clf2.decision_function()[0].tolist()[0]
    print(wgt)
    yaxis = [i for i in range(clf2.num_features)]
    plt.plot(yaxis, wgt)
    plt.show()



if __name__ == "__main__":
  main(sys.argv[1:])
