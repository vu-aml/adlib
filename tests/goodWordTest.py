import sys
from sklearn import svm
from sklearn import linear_model
import adversaries as ad
from data_reader import input, output, extractor
from classifier import Classifier
from metrics import EvasionMetrics
from copy import deepcopy
# debugging
import time


def main(argv):
    """
    driver class that performs attack with the good word attack
    against a logistic regression classfier (aka maxent)
    """

    # create an naive classifier using classifier
    learning_model = linear_model.LogisticRegressionCV()
    instances = input.load_instances('./data_reader/data/test/100_instance_debug')
    clf = Classifier(learning_model, instances)
    clf.train()

    # launch attack on the classifier using a AdversarialStrategy

    adv = ad.GoodWord()
    adv.set_params({'n': 100, 'attack_model_type': 'first_n'})
    adv.set_adversarial_params(clf, instances)
    instances_post_attack = adv.attack(instances)

    # obtain metrics of classification performance both pre- and post- attack
    metrics = EvasionMetrics(clf, instances, instances_post_attack)
    preAccuracy = metrics.getAccuracy(True)
    postAccuracy = metrics.getAccuracy(False)
    print ("Pre-attack Accuracy (naive classifier): " + str(preAccuracy))
    print ("Post-attack Accuracy: (naive classifier): " + str(postAccuracy))

    # plot ROC curve of naive classifier performance after attack
    # currently the AUC is 1 because false positive rate is always in the demo case
    metrics.plotROC(False)

    # create a robust classifier(classifier with a internal mechanism to protect against attack)
    instances2 = input.load_instances('./data_reader/data/test/100_instance_debug')
    learning_model2 = linear_model.LogisticRegressionCV()
    clf2 = Classifier(learning_model2, instances2, ['retraining', ad.SimpleOptimize()])
    clf2.set_simulated_adversary_params({'lambda_val': -100, 'max_change': 65})
    clf2.train()

    # launch attack on the robust classifier using a AdversarialStrategy

    adv = ad.GoodWord()
    adv.set_params({'n': 100, 'attack_model_type': 'first_n'})
    adv.set_adversarial_params(clf, instances)
    instances_post_attack2 = adv.attack(instances2)

    # obtain metrics of classification performance both pre- and post- attack
    metrics2 = EvasionMetrics(clf2, instances, instances_post_attack2)
    preAccuracy2 = metrics2.getAccuracy(True)
    postAccuracy2 = metrics2.getAccuracy(False)
    print ("Pre-attack Accuracy (robust classifier): " + str(preAccuracy2))
    print ("Post-attack Accuracy: (robust classifier): " + str(postAccuracy2))
    metrics.plotROC(True)



if __name__ == "__main__":
  main(sys.argv[1:])
