from data_reader.dataset import EmailDataset
from data_reader.binary_input import Instance
from data_reader.real_input import RealFeatureVector
from data_reader.operations import load_dataset,sparsify
from scipy.sparse import csr_matrix
import numpy as np

dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/full',binary= False,raw=True)
testing_data = load_dataset(dataset)


#print(sparsify(testing_data)[1].toarray())
#print("\n\n\n\n\n\n")



a = testing_data[0].get_feature_vector()
temp = a.get_csr_matrix()

#zero = csr_matrix((1, 909), dtype=np.int8)
#data = zero.data
#indices = zero.indices
#indptr = zero.indptr

#test = RealFeatureVector(909,indices,data)


indptr = np.array([0, 6])
indices = np.array([0, 1,2 ,3 , 4, 5])
data = np.array([1, 2, 3, 4, 5, 6])
zero2 = csr_matrix((data, indices, indptr), shape=(1, 909))

data = zero2.data
indices = zero2.indices
indptr = zero2.indptr

test = RealFeatureVector(909,indices,data)


b = testing_data[1].get_feature_vector()
print(b)

a.flip_val(3,20)

print(a.get_csr_matrix().toarray())
print('\n')
print(b.get_csr_matrix().toarray())
print('\n')
print(a.feature_difference(b).toarray())

#print(a.get_csr_matrix().toarray()[0][0:10])

