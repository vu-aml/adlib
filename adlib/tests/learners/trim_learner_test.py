# trim_learner_test.py
# Tests the TRIM learner implementation
# Matthew Sedam

from adlib.tests.learners.dp_learner_test import TestDataPoisoningLearner
from data_reader.dataset import EmailDataset
import numpy as np
import sys


# TODO: Interpret result


def test_trim_learner():
    if len(sys.argv) == 2 and sys.argv[1] in ['label-flipping', 'k-insertion',
                                              'data-modification', 'dummy']:
        attacker_name = sys.argv[1]
    else:
        attacker_name = 'dummy'

    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',
                           binary=False, raw=True)

    tester = TestDataPoisoningLearner('trim', attacker_name, dataset)
    result = tester.test()

    true_labels = np.array(result[0])
    before_svm_labels = np.array(result[1])
    after_svm_labels = np.array(result[2])
    after_learner_labels = np.array(result[3])
    time = result[4]

    before_svm_incorrect = int((np.linalg.norm(true_labels - before_svm_labels) ** 2) / 4)
    after_svm_incorrect = int((np.linalg.norm(true_labels - after_svm_labels) ** 2) / 4)
    after_learner_incorrect = int((np.linalg.norm(true_labels - after_learner_labels) ** 2) / 4)

    before_svm_percent_correct = (len(true_labels) - before_svm_incorrect) * 100 / len(true_labels)
    after_svm_percent_correct = (len(true_labels) - after_svm_incorrect) * 100 / len(true_labels)
    after_learner_percent_correct = ((len(true_labels) - after_learner_incorrect) * 100 /
                                     len(true_labels))

    print('\n###################################################################')
    print('Before attack SVM correct percentage:', round(before_svm_percent_correct, 4), '%')
    print('After attack SVM correct percentage:', round(after_svm_percent_correct, 4), '%')
    print('After attack learner correct percentage:', round(after_learner_percent_correct, 4), '%')
    print('Elapsed learner time:', round(time, 4), 's')
    print('###################################################################\n')


if __name__ == '__main__':
    test_trim_learner()
