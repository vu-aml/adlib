# dp_learner_many_test.py
# Used to generate many tests of the dp learners
# Matthew Sedam. 2018

import subprocess
import sys


def dp_learner_many_test():
    """
    Used to test the data poisoning learners.
    Use like - python3 adlib/tests/learners/dp_learner_many_test.py 30 label-flipping
    to run 30 tests of label flipping, writing the results to the CWD.
    """
    
    num_runs = int(sys.argv[1]) if len(sys.argv) >= 2 else 30
    attacker = sys.argv[2].lower() if len(sys.argv) == 3 else 'dummy'

    for i in range(num_runs):
        print('START run:', i + 1)

        command = ['python3', 'adlib/tests/learners/dp_learner_test.py', attacker]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = result.stdout.decode('utf-8')

        file_name = './dp-' + attacker + '-' + str(i + 1) + '.txt'
        with open(file_name, 'w+') as file:
            file.write(result)
            file.flush()

        print('END run:', i + 1)


if __name__ == '__main__':
    dp_learner_many_test()
