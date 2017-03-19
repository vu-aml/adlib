import sys
from sklearn import svm
from data_reader import input
from sklearn import metrics
import learners as learner
import random
from data_reader.operations import sparsify
import matplotlib.pyplot as plt

def main(argv):
    """
    driver class that performs demo of the library
    """

    # pre-process data and randomly partition
    instances = input.load_instances('./data_reader/data/test/100_instance_debug')
    random.shuffle(instances)
    instances2 = instances[:60]
    instances3 = instances[60:]
    print(sparsify(instances)[1].toarray())


    # initialize sklearn model
    learning_model = svm.SVC(probability=True, kernel='linear')

    # initialize and train RobustLearner
    clf2 = learner.FeatureDeletion({'hinge_loss_multiplier': 1,
                                    'max_feature_deletion': 50}, instances2)
    clf2.train()

    # produce simple metrics
    y_predict = clf2.predict(instances3)
    y_true = sparsify(instances3)[0]
    score = metrics.precision_score(y_predict,y_true)
    print("score = "+str(score))

    wgt = clf2.decision_function()[0].tolist()[0]
    print(wgt)
    yaxis = [i for i in range(clf2.num_features)]
    plt.plot(yaxis, wgt)
    plt.axis([0,1000,-1,1])
    plt.show()




if __name__ == "__main__":
  main(sys.argv[1:])
