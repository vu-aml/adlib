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

    def __init__(self, num_features: int, feature_indices: List[int]):
        """Create a feature vector given a set of known features.

        Args:
                num_features (int): Total number of features.
                feature_indices (List[int]): Indices of each feature present in instance.

                """
        self.indptr = [0, len(feature_indices)]    # type: List[int]
        self.feature_count = num_features                # type: int
        self.data = [1] * len(feature_indices)     # type: List[int]
        self.indices = feature_indices                     # type: List[int]

    def copy(self, feature_vector):
        return FeatureVector(feature_vector.feature_count, feature_vector.indices)

    def __iter__(self):
        return iter(self.indices)
    
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
            return 1
        else:
            return 0

    def add_feature(self, index, feature):
        """Add feature at given index.

        Switches the current value at the index to the specified value.

                Args:
                        index (int): Index of feature update.
                        feature (int): Boolean, (0/1)

                """
        if feature == 0:
            # should this be -=?
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
        data = [1] * len(self.indices)
        indices = self.indices
        indptr = [0, len(self.indices)]
        return csr_matrix((data, indices, indptr), shape=(1, self.feature_count))
    
    def feature_difference(self, xa) -> List:
        y_array = self.get_csr_matrix()
        xa_array = xa.get_csr_matrix()

        C_y = (y_array - xa_array).indices

        return C_y

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
        self.label = label                                        # type: int
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

def load_instances(data: List) -> List[Instance]:
    """Load data from a specified file.

    Args:
            data (List[str]):

                    data[0]: Data set name.
                    data[1]: Category path (train or test).

    Returns:
            instances as List[Instance]

        """
    path = './data_reader/data/' + data[1] + '/' + data[0]

    instances = []
    max_index = 0
    try:
        with open(path, 'r') as infile:
            for line in infile:
                line = line.replace(',', '')
                instance_data = line.split(' ')
                if '\n' in instance_data[0]:
                    break
                label = int(float(instance_data[0].strip(':')))
                index_list = []
                for feature in instance_data[1:]:
                    if feature == '\n':
                        continue
                    index_list.append(int(feature))
                if index_list[-1] > max_index:
                    max_index = index_list[-1]
                instances.append((label, index_list))

    except FileNotFoundError:
        return None

    num_indices = max_index + 1

    created_instances = []
    for instance in instances:
        feature_vector = FeatureVector(num_indices, instance[1])
        created_instances.append(Instance(instance[0], feature_vector))

    return created_instances


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

