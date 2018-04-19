from typing import List, Dict
from scipy.sparse import csr_matrix, dok_matrix
from data_reader.dataset import EmailDataset
from data_reader.real_input import RealFeatureVector

"""
Created Binary FeatureVector and Instance data structures.
Support converting the emaildataset object(the csr_matrix) into list of instances.
"""


class BinaryFeatureVector(object):
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
        return BinaryFeatureVector(feature_vector.feature_count, feature_vector.indices)

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
                self.indices.remove(index)
                self.indptr[1] -= 1
                self.data.remove(1)
        if feature == 1:
            if index in self.indices:
                return
            else:
                self.indices.append(index)
                self.indices.sort(reverse=True)
                self.indptr[1] += 1
                self.data.append(1)

    def flip_bit(self, index):
        """Flip feature at given index.

        Switches the current value at the index to the opposite value.
        {0 --> 1, 1 --> 0}

                Args:
                        index (int): Index of feature update.

                """
        if index in self.indices:
            self.indices.remove(index)
            self.indptr[1] -= 1
            self.data.remove(1)
        else:
            self.data.append(1)
            self.indices.append(index)
            self.indptr[1] += 1
            self.indices.sort(reverse=True)

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

    def __init__(self, label: int, feature_vector):
        """Create an instance from an existing feature vector.

        Args:
                label (int): Classification (-1/1).
                feature_vector (BinaryFeatureVector): Underlying sparse feature representation.

                """
        self.label = label  # type: int
        self.feature_vector = feature_vector

    def get_label(self):
        return self.label

    def set_label(self, val):
        self.label = val

    def get_feature_vector(self):
        """Return underlying feature vector.

                """
        return self.feature_vector


    def get_feature_count(self):
        """
        :return: Number of features in the underlying feature vector
        """
        return self.feature_vector.get_feature_count()


    def get_csr_matrix(self):
        """
        :return: csr_matrix of the underlying feature vector
        """
        return self.feature_vector.get_csr_matrix()


    def get_feature_vector_cost(self, goal_instance):
        """
           Get the feature differences between two instances.
           Sum all the values up.
        :param goal_vector:
        :return:  a val indicating the differences
        """
        feature_difference = self.get_feature_vector().feature_difference(goal_instance.get_feature_vector())
        sum = 0
        for index in range(len(feature_difference)):
            sum += abs(feature_difference[index])
        return sum


    def flip(self,index,value):
        """
          Chnange the bit at given index
        :param index:
        :param value:
        :return:
        """
        if type(self.get_feature_vector()) == RealFeatureVector:
            self.get_feature_vector().flip_val(index,value)
        else:
            self.get_feature_vector().change_bit(index,value)

