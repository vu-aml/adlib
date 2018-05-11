from typing import Dict


class BaseModel(object):
	"""Abstract base class for learner model

	Defines necessary operations for the underlying
	learner model; training, prediction, classification
	probabilities, and decision function results. In order
	for an initial or improved learner to use a future-defined
	model, the model must include these operations

    """

	def train(self, instances):
		"""Train on the set of training instances.

        Returns:
            None.

        """
		raise NotImplementedError

	def predict(self, instances):
		"""Predict classification labels for the set of instances.

        Returns:
            label classifications (List(int))

        """
		raise NotImplementedError

	def predict_proba(self, instances):
		"""Use the model to determine probability of adversarial classification.

        Returns:
            probability of adversarial classification (List(int))

        """
		raise NotImplementedError

		def predict_log_proba(self, instances):
			"""Use the model to determine probability of adversarial classification.

	        Returns:
	            probability of adversarial classification (List(int))

	        """
			raise NotImplementedError

	def decision_function_(self, instances):
		"""Use the model to determine the decision function for each instance.

        Returns:
            decision values (List(int))

        """
		raise NotImplementedError

	def set_params(self, params: Dict):
		"""Set params for the model.

        Args:
            params (Dict): set of available params with updated values

        """
		raise NotImplementedError

	def get_available_params(self) -> Dict:
		"""Get the set of params defined in the model usage.

        Returns:
            dictionary mapping param names to current values

        """
		raise NotImplementedError

	def get_alg(self):
		"""Return the underlying model algorithm.

        Returns:
            algorithm used to train and test instances

        """
		raise NotImplementedError
