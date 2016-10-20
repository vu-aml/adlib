import sys
import adversarial_environment
from sklearn import svm
from data_reader import extractor

"""This module demonstrates the key library components.

Main components:
  data creation: form training and test data from a given dataset.
  environment creation: start a battle for a given set of data.
  environment loading: load a pre-existing battle.
  parameter querying/setting: get and set params for adversary and learners.
  adversarial operations: initial training, adversary instance manipulation, and improved training.

"""

def create_data():
  dr = extractor.CreateData('demo_data')

  dr.create_corpus('full')
  dr.tf_idf()

  dr.create_corpus('ham25')
  # either create distinct test and train sets, or allow adversary
  # to transform training to test during execution.
  dr.create_instances('all_categories') # test and train instances


def create_new_environment():
  learning_model = svm.SVC(probability=True, kernel='linear')
  test_env = adversarial_environment.Environment('100_instance_debug')
  test_env.create('good_word', learning_model, name='demo', save_state=True)
  return test_env


def load_environment():
  test_env = adversarial_environment.Environment('100_instance_debug')
  test_env.load_environment('demo', save_state=False)
  return test_env


def set_params(test_env):
  params = test_env.get_available_params(adversarial_environment.Environment.ADVERSARY)
  # this doesn't seem very flexible
  # every adversary returns something different for get_available_params
  #params['lambda_val'] = -99
  params['n'] = 200
  test_env.set_params(adversarial_environment.Environment.ADVERSARY, params)


def adversarial_moves(test_env):
  test_env.train_learner()
  test_env.adversary_react(output_transformed_data=True)
  test_env.improve_learner()
  test_env.test_learner(adversarial_environment.Environment.IMPROVED_LEARNER, output_predictions=True)


def main(argv):
  print(argv)
  print('creating enviro')
  #create_data() #: Only uncomment when testing demo of data creation; takes a very long time.
  test_env = create_new_environment()
  print('setting params')
  #test_env = load_environment()
  set_params(test_env)
  print('adversarial moves')
  adversarial_moves(test_env)
  print('done')

if __name__ == "__main__":
  main(sys.argv[1:])
