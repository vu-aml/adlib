from typing import List, Dict
from scipy.sparse import csr_matrix, dok_matrix
from data_reader.dataset import EmailDataset

"""
Created Binary FeatureVector and Instance data structures.
Support converting the emaildataset object(the csr_matrix) into list of instances.
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
        self.indptr = [0, len(feature_indices)]  # type: List[int]
        self.feature_count = num_features  # type: int
        self.data = [1] * len(feature_indices)  # type: List[int]
        self.indices = feature_indices  # type: List[int]

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

    def change_bit(self,index, feature):
        """
        Change the bit only if the feature is different from the current feature
        :param index:
        :param feature:
        """
        if feature == 0:
            if index not in self.indices:
                return
            else:
                self.indices.append(index)
                self.indices.sort()
                self.indptr[1] += 1
                self.data.append(1)
        if feature == 1:
            if index in self.indices:
                return
            else:
                self.indices.remove(index)
                self.indptr[1] -= 1
                self.data.remove(1)

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
        self.label = label  # type: int
        self.feature_vector = feature_vector  # type: FeatureVector

    def get_label(self):
        return self.label

    def set_label(self, val):
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


def load_dataset(emailData: EmailDataset) -> List[Instance]:
    """
    Conversion from dataset object into a list of instances
    :param emailData:
    :return: a list of binary representation of the email data
    """
    instances = []
    num_features = emailData.shape[1]
    indptr = emailData.features.indptr
    indices = emailData.features.indices
    for i in range(0, emailData.num_instances):
        tmp_vector = FeatureVector(num_features, indices[indptr[i]:indptr[i + 1]].tolist())
        instances.append(Instance(emailData.labels[i], tmp_vector))
    return instances
