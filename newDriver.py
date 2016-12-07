import sys
from sklearn import svm
import adversaries.coordinate_greedy
import adversaries.simple_optimize
from data_reader import input, output
from classifier_wrapper import Classifier
from evasion_metrics import EvasionMetrics


def main(argv):
    """
    driver class that performs demo of the library
    """

    # create an naive classifier using classifier_wrapper
    learning_model = svm.SVC(probability=True, kernel='linear')
    clf = Classifier(learning_model, "100_instance_debug")
    clf.train()

    # launch attack on the classifier using a AdversarialStrategy
    instances = input.load_instances(["100_instance_debug", 'test'])
    adv = adversaries.simple_optimize.Adversary()
    adv.set_params({'lambda_val': -100, 'max_change': 65})
    adv.set_adversarial_params(clf, instances)
    instances_post_attack = adv.change_instances(instances)

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
    learning_model2 = svm.SVC(probability=True, kernel='linear')
    clf2 = Classifier(learning_model2, "100_instance_debug", ["retraining", "simple_optimize"])
    clf2.set_simulated_adversary_params({'lambda_val': -100, 'max_change': 65})
    clf2.train()

    # launch attack on the robust classifier using a AdversarialStrategy
    instances2 = input.load_instances(["100_instance_debug", 'test'])
    adv = adversaries.simple_optimize.Adversary()
    adv.set_params({'lambda_val': -100, 'max_change': 65})
    adv.set_adversarial_params(clf2, instances2)
    instances_post_attack2 = adv.change_instances(instances2)

    # obtain metrics of classification performance both pre- and post- attack
    metrics2 = EvasionMetrics(clf2, instances, instances_post_attack2)
    preAccuracy2 = metrics2.getAccuracy(True)
    postAccuracy2 = metrics2.getAccuracy(False)
    print ("Pre-attack Accuracy (robust classifier): " + str(preAccuracy2))
    print ("Post-attack Accuracy: (robust classifier): " + str(postAccuracy2))




if __name__ == "__main__":
  main(sys.argv[1:])
