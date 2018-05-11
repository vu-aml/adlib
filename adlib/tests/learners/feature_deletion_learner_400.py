from sklearn import svm
from learners import SimpleLearner
import learners as learner
from data_reader.dataset import EmailDataset
from data_reader.operations import load_dataset
from learners.feature_deletion import FeatureDeletion
from adversaries.feature_deletion import AdversaryFeatureDeletion
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt

# passed

def summary(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("lengths of two label lists do not match")
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    return "accuracy: {0} \n precision: {1} \n recall: {2}\n".format(acc, prec, rec)


def get_evasion_set(x_test, y_pred):
    # for x, y in zip(x_test, y_pred):
    #     print("true label: {0}, predicted label: {1}".format(x.label, y))
    ls = [x for x, y in zip(x_test, y_pred) if x.label==1 and y==1]
    print("{0} malicious instances are being detected initially")
    return ls, [x.label for x in ls]

dataset = EmailDataset(path='./data_reader/data/raw/trec05p-1/test-400',binary= False,raw=True)
training_, testing_ = dataset.split({'train': 60, 'test': 40})
training_data = load_dataset(training_)
testing_data = load_dataset(testing_)
test_true_label = [x.label for x in testing_data]

# analyzing data feature vector:
# plt.plot(list(range(len(testing_data[2].get_csr_matrix().toarray().reshape((1000,)).tolist()))),
#          testing_data[2].get_csr_matrix().toarray().reshape((1000,)).tolist())
# plt.show()

#test simple learner svm
learning_model = svm.SVC(probability=True, kernel='linear')
learner1 = SimpleLearner(learning_model, training_data)
learner1.train()

# #Print learned weight from initial svm
# print(learner1.get_weight())
# plt.plot(list(range(len(learner1.get_weight()))), testing_data[2].get_csr_matrix().toarray().reshape((1000,)).tolist())
# plt.show()
predictions = learner1.predict(testing_data)
print("======== initial prediction =========")
print(summary(predictions, test_true_label))

# print("finding the set of detected malicious instances for attacking")
# mal_set, mal_label = get_evasion_set(testing_data, predictions)
# print("size of new dataset: {0}".format(len(mal_set)))

# launch attack using feature deletion attacker
attacker = AdversaryFeatureDeletion(num_deletion=40, all_malicious=True)
attacker.set_adversarial_params(learner1, None)
new_testing_data = attacker.attack(testing_data)

# for idx, (x , newx)  in enumerate(zip(testing_data, new_testing_data)):
#     if idx >10:
#         break
#     xarray = x.get_feature_vector().get_csr_matrix().toarray().tolist()[0]
#
#     newxarray = newx.get_feature_vector().get_csr_matrix().toarray().tolist()[0]
#     changed = False
#     num_changed = 0
#     print(newxarray)
#     for index, (num1, num2) in enumerate(zip(xarray, newxarray)):
#         if num1 != num2:
#             print("the {}th feature has been modified, value changed from {} to {}".format(index, num1,num2))
#             changed = True
#             num_changed += 1
#     if changed:
#         print("instance No.{0} is modified. Number of features changed: {1}".format(idx, num_changed))
#         print()

# weight vector from initial learner
w = learner1.model.learner.coef_[0]
b = learner1.model.learner.intercept_[0]
xaxis = range(len(w))
# w1 = learner1.get_weight()
# f, axarr = plt.subplots(2, sharey=True)
# axarr[0].plot(xaxis, w)
# axarr[1].plot(xaxis, w1)
#
# plt.show()

print("verbose prediction")
init_pred_val, init_pred_label = [0]*len(testing_data),[0]*len(testing_data)
atk_pred_val, atk_pred_label = [0]*len(testing_data),[0]*len(testing_data)
# for idx, ins in enumerate(testing_data):
#     x = ins.get_csr_matrix().toarray().tolist()[0]
#     val = np.dot(x, w)+b
#     init_pred_val[idx]= val
#     init_pred_label[idx]=learner1.predict([ins])[0]
#     print ("predict val:  " + str(val))
#     print ("final prediction: " + str(learner1.predict([ins])[0]))
#
# for idx, ins in enumerate(new_testing_data):
#     x = ins.get_csr_matrix().toarray().tolist()[0]
#     val = np.dot(x, w)+b
#     atk_pred_val[idx]= val
#     atk_pred_label[idx]=learner1.predict([ins])[0]
#     print ("post attack predict val:  " + str(val))
#     print ("post attack final prediction: " + str(learner1.predict([ins])[0]))

predictions2 = learner1.predict(new_testing_data)
print("========= post-attack prediction =========")
print(summary(predictions2, test_true_label))

for idx, (p1, p2) in enumerate(zip(init_pred_label, atk_pred_label)):
    if p1 != p2:
        print("Instance {} has successfully evaded, pre-attack value: {}, post attack: {}"
              .format(idx, init_pred_val[idx], atk_pred_val[idx]))


learner2= FeatureDeletion(training_data, params={'hinge_loss_multiplier': 20,
                                    'max_feature_deletion': 10})
learner2.train()
w2 = learner2.get_weight()[0]
b2 = learner2.get_constant()
print("training robust learner...")

print("initial b = {}".format(b))
print("new b = {}".format(b2))
f, axarr = plt.subplots(2, sharey=False)
axarr[0].plot(xaxis, w)
axarr[1].plot(xaxis, w2)

plt.show()


print("========= robust learner post-attack prediction =========")
pred3 = learner2.predict(new_testing_data)
print("true labels")
print(test_true_label)
print("predictions")
print(pred3)
print("number of new prediction: " + str(len(pred3)))
print(summary(pred3, test_true_label))

