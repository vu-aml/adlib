import sys
from sklearn import svm
from data_reader import input
from sklearn import metrics
import learners as learner
import adversaries as ad
import random
from data_reader.operations import sparsify

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
    clf2 = learner.Retraining(learning_model,instances2, {'attack_alg': ad.SimpleOptimize,'adv_params':{}})
    clf2.train()

    # produce simple metrics
    y_predict = clf2.predict(instances3)
    y_true = sparsify(instances3)[0]
    score = metrics.precision_score(y_predict,y_true)
    print("score = "+str(score))



if __name__ == "__main__":
  main(sys.argv[1:])
