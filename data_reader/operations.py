import json, os, pickle
from typing import List, Dict
from scipy.sparse import csr_matrix
from data_reader.binary_input import Instance, BinaryFeatureVector
from data_reader.real_input import RealFeatureVector
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from data_reader.dataset import EmailDataset
from sklearn.ensemble import ExtraTreesClassifier
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

    #num_features = instances[0].get_feature_vector().feature_count
    #indices = []
    #data = []
    #for i in range(num_features):
    #    max = 0
    #    for instance in instances:
    #        value = instance.get_feature_vector().get_feature(i)
    #        if value >= max:
    #            max = value
    #    if max != 0:
    #        indices.append(i)
    #        data.append(max)
    #return RealFeatureVector(num_features, indices, data)


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

    return (labels, csr_matrix((data, indices, indptr), shape=(len(instances), num_features)))


def csr_mat_to_instances(csr_mat, labels, binary= False):
    """
    Return a list of instances
    :param nd_arr:
    :param labels:
    :return:
    """
    data = csr_mat.data
    indices = csr_mat.indices
    indptr = csr_mat.indptr
    instance_len,num_features = csr_mat.shape
    instance_lst = []
    for i in range(instance_len):
        label = labels[i]
        instance_data = data[indptr[i]:indptr[i+1]]
        instance_indices = indices[indptr[i]:indptr[i+1]]
        if binary:
            instance_lst.append(Instance(label, BinaryFeatureVector(num_features, instance_indices)))
        else:
            instance_lst.append(Instance(label, RealFeatureVector(num_features,instance_indices,instance_data)))
    return instance_lst



def nd_arr_to_instances(nd_arr,labels= None, binary= False):
    """
    Return a list of instances
    :param nd_arr:
    :param labels:
    :param binary:
    :return:
    """
    num_instances = nd_arr.shape[0]
    if labels is None:
        labels = nd_arr[:, :1]
        data = nd_arr[:, 1:]
        num_features = nd_arr.shape[1] -1
    else:
        data = nd_arr
        num_features = nd_arr.shape[1]

    instance_lst = []
    for i in range(num_instances):
        if binary:
            mat_indices = [x for x in range(0, num_features) if data[i][x] != 0]
            instance_lst.append(Instance(labels[i],BinaryFeatureVector(num_instances,mat_indices)))
        else:
            mat_indices = [x for x in range(0, num_features) if data[i][x] != 0]
            mat_data = [data[i][x] for x in range(0, num_features) if data[0][x] != 0]
            instance_lst.append(Instance(labels[i], RealFeatureVector(num_instances, mat_indices,mat_data)))
    return instance_lst



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


def feature_selection(instances,max_features ,selection_type = "chi2", binary = False):
    label, sparse_data = sparsify(instances)
    data = sparse_data.toarray()
    if selection_type == "chi2":
        data_new = SelectKBest(chi2, k= max_features).fit_transform(data, label)
        return  nd_arr_to_instances(data_new,label,binary)
    #TODO:ADD Other data selection methods if necessary

