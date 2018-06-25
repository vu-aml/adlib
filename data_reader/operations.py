from typing import List
from scipy.sparse import csr_matrix
from data_reader.binary_input import Instance, BinaryFeatureVector
from data_reader.real_input import RealFeatureVector
from data_reader.dataset import EmailDataset
import math


def fv_equals(fv1, fv2):
    if fv1.indices != fv2.indices:
        return False
    for i in range(len(fv1.data)):
        if fv1.data[i] != fv2.data[i]:
            return False
    return True


def find_centroid(instances: List[Instance]):
    num_features = instances[0].get_feature_vector().feature_count
    indices = []
    data = []
    for i in range(num_features):
        sum = 0
        for instance in instances:
            if instance.label == -1:
                sum += instance.get_feature_vector().get_feature(i)
        sum /= num_features
        if sum != 0:
            indices.append(i)
            data.append(sum)
    return Instance(-1,RealFeatureVector(num_features, indices, data))


def find_max(instances: List[Instance]):
    """
    TODO: find_max and find_min should return an array of length num_features, instead of an object
    :param instances:
    :return:
    """
    num_features = instances[0].get_feature_vector().feature_count
    max_val_list = []
    for i in range(num_features):
        max = 0
        for instance in instances:
            value = instance.get_feature_vector().get_feature(i)
            if value >= max:
                max = value
        max_val_list.append(max)
    return max_val_list



def find_min(instances: List[Instance]):
    num_features = instances[0].get_feature_vector().feature_count
    min_val_list = []
    for i in range(num_features):
        min = 10000
        for instance in instances:
            value = instance.get_feature_vector().get_feature(i)
            if value <= min:
                min = value
        min_val_list.append(min)
    return min_val_list


def sparsify(instances: List[Instance]):
    """
    Return sparse matrix representation of list of instances
    :param instances: binary or real value instances
    :return:
    """
    num_features = instances[0].get_feature_vector().feature_count
    labels = []
    indptr = [0]
    indices = []
    ind_ = 0
    data = []
    for instance in instances:
        labels.append(instance.get_label())
        fv = instance.get_feature_vector()
        indptr.append(indptr[ind_] + len(fv.indices))
        ind_ += 1
        indices += fv.indices
        data.extend(instance.get_feature_vector().data)
    a = len(data)
    b = len(indices)
    return (labels, csr_matrix((data, indices, indptr), shape=(len(instances), num_features)))


def load_dataset(emailData: EmailDataset) -> List[Instance]:
    """
    Conversion from dataset object into a list of instances
    :param emailData:
    """

    instances = []
    num_features = emailData.shape[1]
    indptr = emailData.features.indptr
    indices = emailData.features.indices
    data = emailData.features.data
    for i in range(0, emailData.num_instances):
        if emailData.binary:
            tmp_vector = BinaryFeatureVector(num_features, indices[indptr[i]:indptr[i + 1]].tolist())
        else:
            instance_data = data[indptr[i]:indptr[i + 1]].tolist()
            tmp_vector = RealFeatureVector(num_features, indices[indptr[i]:indptr[i + 1]].tolist(),
                                           instance_data)
        instances.append(Instance(emailData.labels[i], tmp_vector))
    return instances


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    means = mean(numbers)
    variance = sum([pow(x - means, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


def summarize(instances):
    summaries = []
    for i in range(instances[0].get_feature_count()):
        data = []
        for instance in instances:
            data.append(instance.get_feature_vector().get_feature(i))
        summaries.append((mean(data),stdev(data)))
    return summaries
