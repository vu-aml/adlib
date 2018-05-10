# label_flipping_test.py
# Tests the label flipping implementation

from sklearn import svm
from learners import SimpleLearner
import numpy as np
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from adversaries.label_flipping import LabelFlipping


def test_label_flipping():
    # Data processing unit
    # The path is an index of 400 testing samples(raw email data).
    dataset = EmailDataset(path='../../data_reader/data/raw/trec05p-1/test-400',
                           binary=True, raw=True)
    training_data = load_dataset(dataset)

    # Setting the default learner
    # Test simple learner svm
    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, training_data)
    learner.train()

    # Execute the attack
    cost = list(np.random.binomial(2, 0.5, len(training_data)))
    total_cost = 0.3 * len(training_data)  # flip around ~30% of the labels
    attacker = LabelFlipping(learner, cost, total_cost, num_iterations=2,
                             verbose=True)
    attack_data = attacker.attack(training_data)

    flip_vector = []  # 0 -> flipped, 1 -> not flipped
    for i in range(len(attack_data)):
        if attack_data[i].get_label() != training_data[i].get_label():
            flip_vector.append(0)
        else:
            flip_vector.append(1)

    print('Flip vector with 0 -> flipped and 1 -> not flipped: \n',
          np.array(flip_vector))

    original_pred_labels = learner.predict(training_data)

    learning_model = svm.SVC(probability=True, kernel='linear')
    learner = SimpleLearner(learning_model, attack_data)
    learner.train()
    attack_pred_labels = learner.predict(training_data)  # predict w/ orig label

    orig_count = 0
    count = 0
    for i in range(len(training_data)):
        if original_pred_labels[i] != training_data[i].get_label():
            orig_count += 1
        elif attack_pred_labels[i] != training_data[i].get_label():
            count += 1

    orig_precent_correct = ((len(training_data) - orig_count) * 100
                            / len(training_data))
    attack_precent_correct = ((len(training_data) - count) * 100
                              / len(training_data))
    difference = orig_precent_correct - attack_precent_correct

    orig_precent_correct = str(round(orig_precent_correct, 4))
    attack_precent_correct = str(round(attack_precent_correct, 4))
    difference = str(round(difference, 4))
    # round all percentages to 4 decimal places

    print('Original correct percentage: ', orig_precent_correct, '%')
    print('Attack correct percentage: ', attack_precent_correct, '%')
    print('Difference: ', difference, '%')


if __name__ == '__main__':
    test_label_flipping()
