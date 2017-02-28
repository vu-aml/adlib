import json, pickle
from typing import List, Dict
from scipy.sparse import csr_matrix, dok_matrix

"""
Interface between file system and created FeatureVector and Instance data structures.
Some warning about pickle and opening unsafe files.
"""


class FeatureVector(object):
    """Feature vector data structure.

    Contains sparse representation of boolean features.
    Defines basic methods for manipulation and data format changes.

        """
    DEFAULT_WEIGHT = 1

    def __init__(self, num_features: int, feature_indices: List[int], feature_weights = None):
        """Create a feature vector given a set of known features.

        Args:
                num_features (int): Total number of features.
                feature_indices (List[int]): Indices of each feature present in instance.

                """
        self.feature_weights = feature_weights
        self.indices = feature_indices                     # type: List[int]
        self.indptr = [0, len(feature_indices)]    # type: List[int]
        self.feature_count = num_features                # type: int
        self.data = self.get_feature_values()     # type: List[int]

    def copy(self, feature_vector):
        return FeatureVector(feature_vector.feature_count, feature_vector.indices)

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, key):
        return self.indices[key]

    def __len__(self):
      return len(self.indices)

    def get_feature_count(self):
        """Return static number of features.

                """
        return self.feature_count

    def get_feature(self, index: int) -> int:
        """Return value of feature at index

                Args:
                        index (int): Feature index.

                """
        if index in self.indices:
            return self.get_feature_weight(index)
        else:
            return 0

    def add_feature(self, index, feature, weight=DEFAULT_WEIGHT):
        """Add feature at given index.

        Switches the current value at the index to the specified value.

                Args:
                        index (int): Index of feature update.
                        feature (int): Boolean, (0/1)

                """
        if self.feature_weights:
            self.feature_weights[index] = weight

        if feature == 0:
            self.feature_count += 1
            if index in self.indices:
                self.indices.remove(index)
            return
        if feature == 1:
            if index in self.indices:
                return
            self.indices.append(index)
            self.indices.sort()
            self.feature_count += 1
            return

    def remove_feature(self, index):
        """Remove feature at given index.

        If the feature at [index] is 0, no action is taken.

                Args:
                        index (int): Index of feature to remove.

                """
        if index not in self.indices:
            self.feature_count -= 1
        else:
            self.indices.remove(index)
            self.feature_count -= 1

        if self.feature_weights and index in self.feature_weights:
            # TODO: bug here?
            del self.feature_count[index]

    def flip_bit(self, index):
        """Flip feature at given index.

        Switches the current value at the index to the opposite value.
        {0 --> 1, 1 --> 0}

                Args:
                        index (int): Index of feature update.

                """
        if index in self.indices:
            self.indices.remove(index)
        else:
            self.indices.append(index)
            self.indices.sort()

    def get_csr_matrix(self) -> csr_matrix:
        """Return feature vector represented by sparse matrix.

                """
        data = self.get_feature_values()
        indices = self.indices
        indptr = [0, len(self.indices)]
        return csr_matrix((data, indices, indptr), shape=(1, self.feature_count))

    def feature_difference(self, xa) -> List:
        y_array = self.get_csr_matrix()
        xa_array = xa.get_csr_matrix()

        C_y = (y_array - xa_array).indices

        return C_y

    def get_feature_weight(self, index):
        if self.feature_weights and index in self.feature_weights:
            return self.feature_weights[index]
        return FeatureVector.DEFAULT_WEIGHT

    def get_feature_values(self):
        if not self.feature_weights: return [FeatureVector.DEFAULT_WEIGHT for index in self.indices]
        # defaults to 1 if index not in feature_weights
        return [self.feature_weights.get(index, FeatureVector.DEFAULT_WEIGHT) for index in self.indices]

class Instance(object):
    """Instance data structure.

    Container for feature vector and mapped label.

        """

    def __init__(self, label: int, feature_vector: FeatureVector):
        """Create an instance from an existing feature vector.

        Args:
                label (int): Classification (-1/1).
                feature_vector (FeatureVector): Underlying sparse feature representation.

                """
        self.label = label                      # type: int
        self.feature_vector = feature_vector    # type: FeatureVector

    def get_label(self):
        """Return current classification label.

                """
        return self.label

    def set_label(self, val):
        """Set classification to new value.

        Args:
                val (int): New classification label.

                """
        self.label = val
        
    def get_vector(self):
        """Return the underlying data as a list."""
        
        return self.feature_vector.get_feature_values()
    
    def get_feature_vector(self) -> FeatureVector:
        """Return underlying feature vector.

                """
        return self.feature_vector

    # cost of altering feature at given index
    def get_feature_cost(self, cost_vector, index):
        if cost_vector and index in cost_vector:
            return cost_vector[index]
        return 1

    def get_feature_vector_cost(self, goal_vector, cost_vector):
        feature_difference = self.get_feature_vector().feature_difference(goal_vector)
        sum = 0
        for index in feature_difference:
            sum += self.get_feature_cost(cost_vector, index)
        return sum

#TODO: need to change this so that the path to the data is passed rather than
def load_instances(data, continuous=False) -> List[Instance]:
    """Load data from a specified file.

    Args:
            data (str):

                    data[0]: Data set name.
                    data[1]: Category path (train or test).

    Returns:
            instances as List[Instance]

        """
    path = data

    instances = []
    max_index = 0
    try:
        with open(path, 'r') as infile:
            for line in infile:
                line = line.replace(',', '')
                instance_data = line.split(' ')
                if '\n' in instance_data[0]:
                    break
                label, index_list, num_index = read_instance_from_line(instance_data, continuous)
                instances.append((label, index_list))
                max_index = max(num_index,max_index)

    except FileNotFoundError:
        return None

    corpus_weights = None
    if continuous:
        try:
            path += '_corpus_weights'
            with open(path, 'rb') as infile:
                corpus_weights = pickle.load(infile)

        except FileNotFoundError:
            print('Corpus weights file not found')
            return None

    num_indices = max_index + 1
    created_instances = []
    for instance in instances:
        feature_vector = FeatureVector(num_indices, instance[1], corpus_weights)
        created_instances.append(Instance(instance[0], feature_vector))

    return created_instances

def read_instance_from_line(instance_data, continuous):
    label = int(float(instance_data[0].strip(':')))
    max_index = 0
    index_list = []
    for feature in instance_data[1:]:
        if feature == '\n':
            continue
        index_list.append(int(feature))
    if index_list[-1] > max_index:
        max_index = index_list[-1]
    return (label, index_list, max_index)

# def open_transformed_instances(battle_name: str, data: str) -> List[Instance]:
#     path = './data_reader/data/transformed/' + data + '.' + battle_name
#     with open(path, 'r') as infile:
#         instances = json.load(infile)
#     return instances


def open_battle(battle_name: str):
    """Load in saved battle.

    Args:
            battle_name (str): User-specified name of battle.

        """
    path = './data_reader/data/battles/' + battle_name
    with open(path, 'rb') as infile:
        battle = pickle.load(infile)
    return battle


def open_predictions(battle_name: str, data: str) -> List:
    """Load Learner predictions.

    Args:
            battle_name (str): User-specified name of battle.
            data (str): dataset used to generate predictions.

        """
    path = './data_reader/data/predictions/' + data + '.' + battle_name
    with open(path, 'r') as infile:
        predictions = json.load(infile)
    return predictions
