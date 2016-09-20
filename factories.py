from pydoc import locate

"""Factories used to find Models, Initial Learners, Improved Learers, and Adversaries.

	Given the filename of the specified algorithm, returns an instance of the
	object as defined in the file.
"""

class BaseFactory(object):
	LEARNER = 'Learner'
	IMPROVED_LEARNER = 'ImprovedLearner'
	ADVERSARY = 'Adversary'
	MODEL = 'Model'

	def __init__(self, path):
		self.path = path
		self.class_name = None
		self.class_instance = None

	def find_class(self, class_type):
		self.class_name = locate(self.path + class_type)

	def init_class(self):
		self.class_instance = self.class_name()

	def load(self, class_type):
		self.find_class(class_type)
		if self.class_name is not None:
			self.init_class()

	def get_class(self):
		return self.class_instance


class ModelFactory(BaseFactory):
	def __init__(self, model_alg):
		BaseFactory.__init__(self, 'learners.models.' + model_alg + '.')
		self.type = BaseFactory.MODEL

	def load_class(self):
		BaseFactory.load(self, self.type)


class InitialPredictorFactory(BaseFactory):
	def __init__(self, learner_model_alg):
		BaseFactory.__init__(self, 'learners.' + learner_model_alg + '.')
		self.type = BaseFactory.LEARNER

	def load_class(self):
		BaseFactory.load(self, self.type)


class ImprovedPredictorFactory(BaseFactory):
	def __init__(self, learner_improve_alg):
		BaseFactory.__init__(self, 'learners.' + learner_improve_alg + '.')
		self.type = BaseFactory.IMPROVED_LEARNER

	def load_class(self):
		BaseFactory.load(self, self.type)


class AdversaryFactory(BaseFactory):
	def __init__(self, adversary_alg):
		BaseFactory.__init__(self, 'adversaries.' + adversary_alg + '.')
		self.type = BaseFactory.ADVERSARY

	def load_class(self):
		BaseFactory.load(self, self.type)
