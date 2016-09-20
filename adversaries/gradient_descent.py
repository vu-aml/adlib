from adversaries.adversary import AdversaryStrategy
from typing import List, Dict
from data_reader.input import Instance, FeatureVector
import numpy as np
from sklearn.svm import SVC
from copy import deepcopy


'''Gradient descent to find locally optimal adversarial instance.

Concept:
	Uses gradient of learning model decision function to converge on a local minima
	for the mapped value of the adversarial instance.
'''

class Adversary(AdversaryStrategy):

	def __init__(self):
		#currently only supported for sklearn classifier
		self.model = None       # type: SVC
		self.lambda_val = 1.0
		self.max_iteration = 10
		self.step = 1.0
		self.gradient_function = self.loss_function = getattr(self, 'linear_gradient')

	def change_instances(self, instances) -> List[Instance]:
		transformed_instances = []
		for instance in instances:
			transformed_instance = deepcopy(instance)
			if instance.get_label() == 1:
				transformed_instances.append(self.gradient_descent(transformed_instance, instances))
			else:
				transformed_instances.append(transformed_instance)
		return transformed_instances

	def set_params(self, params: Dict):
		if params['lambda_val'] is not None:
			self.lambda_val = params['lambda_val']

		if params['num_iterations'] is not None:
			self.max_iteration = params['num_iterations']

		if params['step'] is not None:
			self.step = params['step']

		if params['gradient_function'] is not None:
			name = params['gradient_function']
			self.gradient_function = getattr(self, name)

	def get_available_params(self) -> Dict:
		params = {'lambda_val': self.lambda_val,
		          'num_iterations': self.max_iteration,
		          'step': self.step,
		          'gradient_function': 'linear_gradient'}
		return params

	def set_adversarial_params(self, learner, train_instances):
		self.model = learner.get_model().get_alg()

	def gradient_descent(self, instance: Instance, instances: List[Instance]):
		features = instance.get_feature_vector().get_csr_matrix().toarray()[0]
		other_features = [(x.get_label(),
		                   x.get_feature_vector().get_csr_matrix().toarray()[0]) for x in instances]
		for i in range(0,self.max_iteration):
			gradient = self.gradient_function(features, other_features)
			dg = np.linalg.norm(gradient)**-1
			gradient = np.multiply(1/np.linalg.norm(gradient), gradient)
			new_features = np.subtract(features, np.multiply(self.step, gradient))
			if self.model.decision_function(new_features)[0] - self.model.decision_function(features)[0] < .1:
				break
			features = new_features

		indices = [x for x in range(0,len(features)) if features[x] == 1]
		transformed_instance = Instance(instance.get_label(), FeatureVector(len(features), indices))
		return transformed_instance

	def linear_gradient(self, features, other_features):
		return self.model.coef_.toarray()

	def rbf_gradient(self, features, other_features):
		alphas = self.model.dual_coef_[0]
		gamma = self.model.get_params()['gamma']
		if gamma is 'auto':
			gamma = 1/len(features)
		sum_ = np.zeros(len(features))
		for val in zip(alphas, other_features):
			alpha = val[0]
			label = val[1][0]
			next_features = val[1][1]
			feature_diff = np.subtract(features, next_features)
			sum_ = np.add(sum_, alpha*label*(np.multiply(-2*gamma*np.exp(-gamma*np.dot(feature_diff, feature_diff)), feature_diff)))
		return sum_


