import json, os, pickle
from typing import List, Dict
from scipy.sparse import csr_matrix
from data_reader.input import Instance, FeatureVector


def fv_equals(fv1: FeatureVector, fv2: FeatureVector):
	return fv1.indices == fv2.indices


def sparsify(instances: List[Instance]):
	num_features = instances[0].get_feature_vector().feature_count
	labels = []
	indptr = [0]
	indices = []
	ind_ = 0
	for instance in instances:
		labels.append(instance.get_label())
		fv = instance.get_feature_vector()
		indptr.append(indptr[ind_]+len(fv.indices))
		ind_ += 1
		indices += fv.indices
	data = [1]*len(indices)
	return (labels, csr_matrix((data, indices, indptr), shape=(len(instances), num_features)))
