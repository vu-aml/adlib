import sys
from sklearn import svm
from data_reader import binary_input
from sklearn import metrics
import learners as learner
import random
from random import seed, shuffle
from data_reader.dataset import EmailDataset
def main(argv):
    """
    driver class that performs demo of the library
    """

    # pre-process data and randomly partition
    dataset = EmailDataset(path='../../data_reader/data/test/100_instance_debug.csv', raw=False)
    # set a seed so we get the same output every time
    seed(1)
    features, label = dataset.numpy()
    print(features.shape)
    print(label.shape)

    # initialize sklearn model
    learning_model = svm.SVC(probability=True, kernel='linear')

    # initialize and train RobustLearner
    clf2 = learner.FeatureDeletion(features[:60], label.reshape(len(label))[:60],{'hinge_loss_multiplier': 1,
                                    'max_feature_deletion': 0})
    clf2.train()

    # produce simple metrics
    y_predict = clf2.predict(features[60:])
    print(y_predict)
    y_true = label[60:]
    print(y_true)
    score = metrics.accuracy_score(y_true, y_predict)
    print("score = "+str(score))





if __name__ == "__main__":
  main(sys.argv[1:])
