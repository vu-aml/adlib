from typing import List
from scipy.sparse import csr_matrix

"""
Created Binary FeatureVector and Instance data structures.
Support converting the emaildataset object(the csr_matrix) into list of instances.
"""


class RealFeatureVector(object):
    """Feature vector data structure.

    Contains sparse representation of real_value features.
    Defines basic methods for manipulation and data format changes.

        """

    def __init__(self, num_features: int, feature_indices: List[int], data):
        """Create a feature vector given a set of known features.

        Args:
                num_features (int): Total number of features.
                feature_indices (List[int]): Indices of each feature present in instance.

                """
        self.indptr = [0, len(feature_indices)]  # type: List[int]
        self.feature_count = num_features  # type: int
        self.indices = feature_indices  # type: List[int]
        self.data = data

    def copy(self, feature_vector):
        return RealFeatureVector(feature_vector.feature_count, feature_vector.indices,
                                 feature_vector.data)

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

    def get_feature(self, index: int):
        """Return value of feature at index
                Args:
                        index (int): Feature index.
                """
        for i in range(len(self.indices)):
            if index == self.indices[i]:
                return self.data[i]
        return 0

    def flip_val(self, index, value):
        if index not in self.indices:
            if value == 0:
                return
            self.indices.append(index)
            self.indices.sort(reverse=True)
            self.indptr[1] += 1
            for i in range(len(self.indices)):
                if index == self.indices[i]:
                    self.data.insert(i, value)
                    return
        else:
            for i in range(len(self.indices)):
                if index == self.indices[i]:
                    if value != 0:
                        self.data[i] = value
                        return
                    else:
                        self.indices.remove(index)
                        self.indptr[1] -= 1
                        self.data.pop(i)
                        return

    def get_csr_matrix(self) -> csr_matrix:

        """Return feature vector represented by sparse matrix.

                """
        data = self.data
        indices = self.indices
        indptr = [0, len(self.indices)]
        return csr_matrix((data, indices, indptr), shape=(1, self.feature_count))

    def feature_difference(self, xa):
        y_array = self.get_csr_matrix()
        xa_array = xa.get_csr_matrix()
        c_y = (y_array - xa_array)
        return c_y.data
