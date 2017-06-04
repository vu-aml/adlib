from data_reader.input import Instance
from typing import List
from learners.learner import RobustLearner
from sklearn import metrics
from data_reader.operations import sparsify
import matplotlib.pyplot as plt


class EvasionMetrics(object):
    """
    Metrics class for classifier performance evaluation
    Current supported metrics:
        precision, recall, accuracy, ROC, area under ROC curve

    """

    def __init__(self, clf: RobustLearner, pre: List[Instance], post: List[Instance]):
        self.classifier = clf
        self.pre_attack_instances = pre     # Type: List[Instance]
        self.post_attack_instances = post   # Type: List[Instance]

    def getPrecision(self, isPreAttack):
        if isPreAttack:
            y_true = self.pre_attack_instances.labels
            y_pred = self.classifier.predict(self.pre_attack_instances.features)
        else:
            y_true = self.post_attack_instances.labels
            y_pred = self.classifier.predict(self.post_attack_instances.features)
        return metrics.precision_score(y_true, y_pred)

    def getRecall(self, isPreAttack):
        if isPreAttack:
            y_true = self.pre_attack_instances.labels
            y_pred = self.classifier.predict(self.pre_attack_instances.features)
        else:
            y_true = self.post_attack_instances.labels
            y_pred = self.classifier.predict(self.post_attack_instances)
        return metrics.recall_score(y_true, y_pred)

    def getAccuracy(self, isPreAttack):
        if isPreAttack:
            y_true = self.pre_attack_instances.labels
            y_pred = self.classifier.predict(self.pre_attack_instances.features)
        else:
            y_true = self.post_attack_instances.labels
            y_pred = self.classifier.predict(self.post_attack_instances.features)
        return metrics.accuracy_score(y_true, y_pred)

    def getROC_AUC(self, isPreAttack):
        if isPreAttack:
            y_true = self.pre_attack_instances.labels
            score = self.classifier.decision_function(self.pre_attack_instances.features)
        else:
            y_true = self.post_attack_instances.labels
            score = self.classifier.decision_function(self.post_attack_instances.features)
        return metrics.roc_auc_score(y_true, score)

    def plotROC(self, isPreAttack):
        plt.figure()
        if isPreAttack:
            y_true = self.pre_attack_instances.labels
            score = self.classifier.decision_function(self.pre_attack_instances)
        else:
            y_true = self.post_attack_instances.labels
            score = self.classifier.decision_function(self.post_attack_instances)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, score)
        roc_auc = metrics.roc_auc_score(y_true, score)
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
