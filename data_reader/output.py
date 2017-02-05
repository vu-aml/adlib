import pickle
from typing import List
from data_reader.input import Instance


def save_transformed_instances(battle_name: str, data: str, instances: List[Instance]):
	"""Save data from a specified file.

	Args:
	    battle_name (str): User-specified name of battle.
	    data (str): Name of dataset that was transformed.
	    instances (List[Instance]): Transformed instances to save.

    """
	path = './data_reader/data/transformed/' + data + '.' + battle_name
	with open(path, 'w') as outfile:
		for instance in instances:
			label = instance.get_label()
			indices = instance.get_feature_vector().indices
			instance_str = str(label).strip('[]') + ': ' + str(indices).strip('[]')
			outfile.write(instance_str+'\n')


def save_data(category: str, name: str, instances: List[List[int]], corpus_weights=None):
    """Save instances extracted from corpus.
    
    Args:
        category (str): Train or Test.
        name (str): User-specified name of dataset.
        instances (List[List[int]): Raw data to save.
    
    """
    path = './data_reader/data/' + category + '/' + name
    with open(path, 'w') as outfile:
    	for instance in instances:
            write_instance_to_file(outfile, instance)
    path += '_corpus_weights'
    write_weights_to_file(path, corpus_weights)

def write_instance_to_file(outfile, instance):
    instance_str = str(instance[0]).strip('[]') + ': ' + str(instance[1:]).strip('[],')
    outfile.write(instance_str+'\n')

def write_weights_to_file(path, weights):
    with open(path, 'wb') as outfile:
        pickle.dump(weights, outfile)

def save_battle(battle, battle_name):
	"""Save battle at a given state of execution.

	Args:
	    battle (Battle): Existing object containing learner and adversary in a given state.
	    battle_name (str): User-specified name of battle.

    """
	path = './data_reader/data/battles/' + battle_name

	with open(path, 'wb') as outfile:
		pickle.dump(battle, outfile, -1)


def save_predictions(battle_name: str, data: str, predictions: List):
	"""Save learner generated predictions.

	Args:
	    battle_name (str): User-specified name of battle.
	    data (str): Name of dataset that was transformed.
	    predictions (List): Predictions for each instance.

    """
	path = './data_reader/data/predictions/' + data + '.' + battle_name
	with open(path, 'w') as outfile:
		for prediction in predictions:
			outfile.write(str(prediction) + '\n')
