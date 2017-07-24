import os
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from scipy.sparse import csr_matrix, dok_matrix, find
import sklearn
import numpy as np
import csv
import pickle
from collections import namedtuple
from copy import copy, deepcopy


class Dataset(object):

    def __init__(self):
        return

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def create_corpus(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

class EmailDataset(Dataset):
    """Dataset which loads data from either raw email txt files, a serialized
    sparse matrix representation of the dataset (either .pkl or .csv), or
    loads preloaded features and labels into an EmailDataset to be used
    with the rest of the library.
    In the case of loading raw email data, `path` can be viewed as the path to
    an index .txt file that contains relative paths to other email txt files.
    In the case of loading a serialized dataset, it specifies the file to a
    .pkl or .csv file containing a sparse representation of the data.
    Args:
        path (str): Path to index file that will be passed to the
            `_create_corpus` function.
        raw (boolean, optional): If true then load corpus from raw email txt.
        features (scipy.sparse.csr_matrix, optional): dataset feature matrix
        labels (numpy.ndarray, optional): dataset labels corresponding to the
            feature matrix.
        binary (boolean, optional): Feature type, continuous (False) by default
    """

    def __init__(self, path=None, raw=True, features=None, labels=None,
                 binary=True, strip_accents_=None, ngram_range_=(1, 1),
                 max_df_=1.0, min_df_=1, max_features_=1000, num_instances = 0):
        super(EmailDataset, self).__init__()
        self.num_instances = num_instances
        self.binary = binary
        if path is not None:
            self.base_path = os.path.dirname(path)
        #: Number of instances within the current corpus
            if raw:
                self.labels, self.corpus = self._create_corpus(path)
            # Sklearn module to fit/transform data and resulting feature matrix
            # Maybe optionally pass this in as a parameter instead.
                self.vectorizer = \
                    TfidfVectorizer(analyzer='word', strip_accents=strip_accents_,
                                    ngram_range=ngram_range_, max_df=max_df_,
                                    min_df=min_df_, max_features=max_features_,
                                    binary=False, stop_words='english',
                                    use_idf=True, norm=None)
                self.vectorizer = self.vectorizer.fit(self.corpus)
                self.features = self.vectorizer.transform(self.corpus)
            else:
                self.labels, self.features = \
                    self._load(path, os.path.splitext(path)[1][1:])
        elif path is None and features is not None and labels is not None:
            lbl = type(labels)
            if lbl != np.ndarray and lbl != np.float64 and lbl != int and lbl != float:
                raise ValueError("Labels must be in the form of a numpy array, a float, or an int")
            assert type(features) == scipy.sparse.csr.csr_matrix

            self.features = features
            self.labels = labels
        else:
            raise AttributeError('Incorrect combination of parameters.')
        self.shape = self.features.shape
        self.Data = namedtuple('EmailDataset', 'features labels')
        self.data = self.Data(self.features, self.labels)

    def _create_corpus(self, folder):
        """Generate list of files, one for each future instance and labels
        for the instances in the corpus.

        Args:
            folder (str): Path specifying %ham and %spam emails.

        Returns:
            labels (numpy.ndarray): Feature vector labels for the dataset.
            corpus (List(str)): List of parsed email text.

        """
        # Reset stored values when making new corpus
        corpus = []
        index_path = folder + '/index'
        labels, files = self._find_files(index_path)
        labels = np.array(labels)

        for file in files:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8', errors='replace',
                          buffering=(2 << 16) + 8) as email:
                    email_string = email.read().replace('\n\t', ' ') \
                        .replace('\n', ' ')
                    corpus.append(email_string)
        return labels, corpus

    def _find_files(self, index_path):
        """Generate list of file paths, one for each future instance.

        Args:
            index_path (str): Path containing file that specifies locations of
            raw files

        Returns:
            file paths (List(str))

        """
        files = list()
        labels = list()
        with open(index_path, 'r', buffering=(2 << 16) + 8) as file_list:
            for line in file_list:
                category_path = line.replace('\n', '').split(' ../')
                filepath = os.path.join(self.base_path, category_path[1])
                if os.path.isfile(filepath):
                    if category_path[0] == 'spam':
                        labels.append(1)
                    else:
                        labels.append(-1)
                    files.append(filepath)
                    self.num_instances += 1
        return labels, files

    def __getitem__(self, index):
        if type(index) == tuple:
            if len(index) > 2:
                raise ValueError("Email Datasets only support two dimensions.")
            else:
                # maybe return emaildataset instance with corresponding label?
                return self.features[index]
        return self.Data(features=self.features[index], labels=self.labels[index])

    def __setitem__(self, index, value):
        self.features[index] = value.features
        self.labels[index] = value.labels

    def __len__(self):
        return self.features.shape[0]

    def __str__(self):
        return str(self.data)

    def index(self, index, sparse=False):
        """Method that can be used to index the dataset with the option of
            a sparse or dense representation.

        Args:
            index (int): Index of dataset to return.
            sparse (boolean, optional): If True, return sparse feature matrix,
                else return `numpy.ndarray` representation. Default: False

        Returns:
            instance (namedtuple(features, labels)): Return either a sparse
                or dense instance from the dataset.

        """
        if sparse:
            return self.Data(self.features[index], self.labels[index])

        else:
            return self.__getitem__(index)

    def numpy(self):
        """This is a convenience method for __getitem__[:]"""
        return self.Data(self.features.toarray(), self.labels)

    def sort(self):
        """Sort the features in place by index"""
        self.features.sort_indices()

    def __eq__(self, other):
        if isinstance(other, EmailDataset):
            if self.features.shape == other.features.shape and self.features.dtype == other.features.dtype:
                if (self.features != other.features).nnz() == 0:
                    return self.labels == other.labels
        else:
            return False

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def clone(self):
        """Return a new copy of the dataset with same initial params."""

        return self.Data(features=self.features.copy(), labels=self.labels.copy())

    def _csv(self, outfile, save=True):
        # load a .csv file where all the data in the file mark the relative postions,
        # not values if save = true, save [[label, *features]] to standard csv file
        if save:
            with open(outfile, 'w') as fileobj:
                serialize = csv.writer(fileobj)
                data = np.concatenate((self.labels[:, np.newaxis],
                                       self.features.toarray()), axis=1)
                for instance in data.tolist():
                    serialize.writerow(instance)
        else:
            # TODO: throw exception if FileNotFoundError
            data = np.genfromtxt(outfile, delimiter=',')
            self.num_instances = data.shape[0]
            labels = data[:, :1]
            feats = data[:, 1:]
            mask = ~np.isnan(feats)
            col = feats[mask]
            row = np.concatenate([np.ones_like(x)*i
                                 for i, x in enumerate(feats)])[mask.flatten()]
            features = csr_matrix((np.ones_like(col), (row, col)),
                                  shape=(feats.shape[0],
                                  int(np.max(feats[mask]))+1))
            return np.squeeze(labels), features

    def _pickle(self, outfile, save=True):
        """A fast method for saving and loading datasets as python objects.

        Args:
            outfile (str): The destination file.
            save (boolean, optional): If True, serialize, if False, load.

        """
        if save:
            with open(outfile, 'wb') as fileobj:
                pickle.dump({
                            'labels': self.labels,
                            'features': self.features
                            }, fileobj, pickle.HIGHEST_PROTOCOL)
        else:
            # TODO: throw exception if FileNotFoundError
            with open(outfile, 'rb') as fileobj:
                data = pickle.load(fileobj)
                return data['labels'], data['features']

    def save(self, outfile='~/data/serialized.pkl', binary=False):
        """User facing function for serializing an `EmailDataset`.

        Args:
            outfile (str, optional): The destination file.
            binary(boolean, optional): If True, save as binary sparse
                representation.

        """
        format = os.path.splitext(outfile)[1][1:]
        if format == 'csv':
            self._csv(outfile)
        elif format == 'pkl':
            self._pickle(outfile)
        else:
            raise AttributeError('The given save format is not currently \
                                 supported.')

    def _load(self, path, format='pkl', binary=False):
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
        if format == 'pkl':
            return self._pickle(path, save=False)
        elif format == 'csv':
            return self._csv(path, save=False)
        else:
            raise AttributeError('The given load format is not currently \
                                 supported.')

    def split(self, split={'test': 50, 'train': 50}):
        """Split the dataset into test and train sets using
            `sklearn.utils.shuffle()`.

        Args:
            split (Dict, optional): A dictionary specifying the splits between
                test and trainset.  The values can be floats or ints.

        Returns:
            trainset, testset (namedtuple, namedtuple): Split tuples containing
                share of shuffled data instances.

        """
        splits = list(split.values())
        for s in splits:
            if s < 0:
                raise ValueError('Split percentages must be positive values')
        # data = self.features.toarray()
        frac = 0
        if splits[0] < 1.0:
            frac = splits[0]
        else:
            frac = splits[0]/100
        pivot = int(self.__len__()*frac)
        s_feats, s_labels = sklearn.utils.shuffle(self.features, self.labels)
        print(type(s_feats))
        print(type(s_labels))
        return (self.__class__(raw=False, features=s_feats[:pivot, :],
                               labels=s_labels[:pivot],num_instances = pivot,binary= self.binary),
                self.__class__(raw=False, features=s_feats[pivot:, :],
                               labels=s_labels[pivot:],num_instances = self.num_instances - pivot,binary= self.binary))
        # return (self.Data(s_feats[:pivot, :], s_labels[:pivot]),
        #         self.Data(s_feats[pivot:, :], s_labels[pivot:]))
