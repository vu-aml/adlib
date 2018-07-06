# trim_learner_test.py
# Tests the TRIM learner implementation
# Matthew Sedam

from adlib.tests.learners.dp_learner_test import TestDataPoisoningLearner
from adlib.utils.common import report
from data_reader.dataset import EmailDataset
import sys


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
    report(result)


if __name__ == '__main__':
    test_trim_learner()
