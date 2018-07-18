import pickle, json
from typing import List
from data_reader.binary_input import Instance
from scipy.sparse import csr_matrix, dok_matrix, find
import os
import csv
import pickle
import numpy as np
from data_reader.operations import sparsify, csr_mat_to_instances


def save(data, outfile='./data_reader/data/transformed/serialized.pkl', binary=False):
    """User facing function for serializing an instance object.

    Args:
        outfile (str, optional): The destination file.
        binary(boolean, optional): If True, save as binary sparse
            representation.

    """

    format = os.path.splitext(outfile)[1][1:]
    if format == 'csv':
        _csv(outfile, save=True, data=data, binary=binary)
    elif format == 'pkl':
        _pickle(outfile, save=True, data=data, binary=binary)
    else:
        raise AttributeError('The given save format is not currently \
                               supported.')


def load(path, binary=False):
    """Load function called by `__init__()` if path is specified and
        `raw = False`.

    Args:
        path (str): Path to load serialized sparse dataset from.
        format (str, optional): Either pkl or csv. Default: pkl

    Returns:
        labels (np.ndarray): The labels for loaded dataset.
        features (scipy.sparse.csr_matrix): The sparse feature matrix of
            loaded dataset.

    """
    format = os.path.splitext(path)[1][1:]
    if format == 'pkl':
        return _pickle(path, save=False, binary=binary)
    elif format == 'csv':
        return _csv(path, save=False, binary=binary)
    else:
        raise AttributeError('The given load format is not currently \
                                 supported.')


def _csv(outfile, binary, save=True, data=None):
    # data: instances
    # load a .csv file where all the data in the file mark the relative postions,
    # not values if save = true, save [[label, *features]] to standard csv file
    if save:
        label, sparse_data = sparsify(data)
        with open(outfile, 'w+') as fileobj:
            serialize = csv.writer(fileobj)
            data = np.concatenate((np.array(label)[:, np.newaxis],
                                   sparse_data.toarray()), axis=1)
            for instance in data.tolist():
                serialize.writerow(instance)
    else:
        # TODO: throw exception if FileNotFoundError
        data = np.genfromtxt(outfile, delimiter=',')
        num_instances = data.shape[0]
        labels = data[:, :1]
        feats = data[:, 1:]
        features = csr_matrix(feats)
        if binary:
            return csr_mat_to_instances(features, np.squeeze(labels), binary=True)
        else:
            return csr_mat_to_instances(features, np.squeeze(labels), binary=False)


def _pickle(outfile, binary, save=True, data=None):
    """A fast method for saving and loading datasets as python objects.

    Args:
        outfile (str): The destination file.
        save (boolean, optional): If True, serialize, if False, load.

    """
    if save:
        label, sparse_data = sparsify(data)
        with open(outfile, 'wb+') as fileobj:
            pickle.dump({
                'labels': label,
                'features': sparse_data
            }, fileobj, pickle.HIGHEST_PROTOCOL)
    else:
        # TODO: throw exception if FileNotFoundError
        with open(outfile, 'rb') as fileobj:
            data = pickle.load(fileobj)
            if binary:
                return csr_mat_to_instances(data['features'], data['labels'], binary=True)
            else:
                return csr_mat_to_instances(data['features'], data['labels'], binary=False)
