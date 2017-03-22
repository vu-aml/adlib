.. _Extending-AdLib:

Extending AdLib
===============

Here we'll cover ways of extending :mod:`aml.adversaries` and :mod:`aml.learners`


Extending :mod:`aml.adversaries`
--------------------------------

.. currentmodule:: adversaries

:mod:`~aml.adversaries` exports a simple interface that all aml adversaries must
inherit.

Adding an :class:`Adversary`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Since :mod:`~aml.adversaries` relies on the abstract base class :class:`Adversary`,
adding a new :class:`Adversary` requires implementing 2 simple functions:

- ``__init__`` - Takes in arguments such as lambda value, corresponding
  :class:`RobustLearner`,number of features, etc. and initializes parameters.
  of features, etc. and initializes parameters and buffers.
- :meth:`~Adversary.attack` - Performs the bulk of the classes'
  utility, as it defines a specific algorithm for modifying training instances
  that the :class:`RobustLearner` will use for training.  This function can call
  other helper functions if you prefer.

Note: One can also optionally implement the following 3 methods.  They are recommended
if the parameters of the :class:`Adversary` will be manipulated frequently by the user.

- :meth:`~Adversary.set_params` - Simply initializes the parameters
  by passing them in a dictionary.
- :meth:`~Adversary.set_adversarial_params` - Initializes the parameters that
  are specific to the adversary that should not be changed after initialization
  (e.g. the learning model or the number of features).
- :meth:`~Adversary.get_available_params` - Getter function that returns the
  params set using :meth:`~Adversary.set_params` or those that were initialized
  by the constructor but are not included in :mod:`adversarial_params`.


For example, this is how the ``SimpleOptimize`` adversary can be implemented::

    class SimpleOptimize(Adversary):
        def __init__(self, lambda_val=-100,max_change=1000,learner=None):
              Adversary.__init__(self)
              self.lambda_val = lambda_val                     # type: float
              self.max_change = max_change                     # type: float
              self.num_features = None                         # type: int
              self.learn_model = learner

        def attack(self, instances: List[Instance]) -> List[Instance]:
            # The attack function is called by passing a list containing
            # :class:`Instance`s.  It is typical to loop over instances and
            # perform the attack on each one individually.  Here we call a helper
            # function to optimize the distance that we move the features within
            # the feature space.  See the SimpleOptimize section for explanation
            # of this specific algorithm.
            transformed_instances = []
            if self.num_features is None:
                self.num_features = instances[0].get_feature_vector().get_feature_count()
            for instance in instances:
                transformed_instance = deepcopy(instance)
                if instance.get_label() == 1:
                    transformed_instances.append(self.optimize(transformed_instance))
                else:
                    transformed_instances.append(transformed_instance)
            return transformed_instances

        def set_params(self, params: Dict):
            if 'lambda_val' in params.keys():
                self.lambda_val = params['lambda_val']
            if 'max_change' in params.keys():
                self.max_change = params['max_change']

        def get_available_params(self) -> Dict:
            params = {'lambda_val': self.lambda_val,
                      'max_change': self.max_change}
            return params

        def set_adversarial_params(self, learner, train_instances: List[Instance]):
            self.learn_model = learner
            self.num_features = train_instances[0].get_feature_vector().get_feature_count()

        def optimize(self, instance: Instance):
            # Here you can see we define a helper function to perform the attack
            # on the instances.  See the SimpleOptimize section for explanation
            # of this specific algorithm.
            change = 0
            for i in range(0, self.num_features):
                orig_prob = self.learn_model.predict_proba([instance])[0]
                instance.get_feature_vector().flip_bit(i)
                change += 1
                new_prob = self.learn_model.predict_proba([instance])[0]
                if new_prob >= (orig_prob-exp(self.lambda_val)):
                    instance.get_feature_vector().flip_bit(i)
                    change -= 1
                if change > self.max_change:
                    break
            return instance



Extending :mod:`aml.learners`
-----------------------------

.. currentmodule:: learners

Adding learning algorithms to :mod:`~aml.learners` requires implementing a new
:class:`RobustLearner` subclass for each learner algorithm. Every new learner
requires you to implement 5 methods:

- ``__init__`` - if your learner is parametrized by/uses a specific model, or
  specific constraint values for instance, you should pass them as arguments
  to ``__init__``. For example, the ``SVMRestrained`` learner takes a c_delta
  parameter that quantifies the tradeoff between camouflage and maximal feature
  value change, while ``SimpleLearner`` requires specifying which base model
  the algorithm will use.
- :meth:`~RobustLearner.set_training_instances` (*optional*) - simply initializes
  the set of training instances that the learner will train on.  If you prefer to
  pass the instances after initialization of the learner than you can implement this
  method, but the training instances must be passed in to the learner before any
  training can occur.
- :meth:`~RobustLearner.train` - the code that performs the training of the learner.
  This method does not take any arguments as it should perform the bulk of the learning
  algorithm using only instance variables and variables assigned within the function.
  It's primary function should be to train on the set of training instances
  that were passed at initialization or by :meth:`~RobustLearner.set_training_instances`.
- :meth:`~RobustLearner.set_params` - Similar to the adversary equivalent, this
  just allows one to pass in a dictionary of parameters after initialization of
  the :class:`RobustLearner`.
- :meth:`~RobustLearner.get_available_params` - Similar to the adversary equivalent,
  this getter function allows one to retrieve a dictionary of learner parameters.
- :meth:`~RobustLearner.predict` - Code specifying method for predicting classification
  labels for the set of instances.
- :meth:`~RobustLearner.predict_proba` (*optional*) - Provides an sklearn-esque way
  of returning class probabilities over a set of instances rather than the labels.
- :meth:`~RobustLearner.decision_function` - Returns learned weight matrix and bias
  vector.

Below you can find code for a ``SimpleLearner`` class from :mod:`aml.learners`, with
additional comments::


    class SimpleLearner(RobustLearner):
        """Simple Learner for initial learning methods.
        Defines the bare-minimum functionality for initial learning
        strategies.
        """

        def __init__(self, model = None, training_instances = None):
            RobustLearner.__init__(self)
            # set model to use as base model
            # set the feature instances to be used in training
            if model: self.set_model(model)
            else: self.model = None
            self.training_instances = training_instances

        def set_model(self, model):
            # helper function to wrap model with sklearner model wrapper
            self.model = sklearner.Model(model)

        def train(self):
            # optionally pass a list of instances or a single instance
            if not self.model:
                raise ValueError('Must specify classification model')
            if not self.training_instances:
                raise ValueError('Must set training instances before training')
            self.model.train(self.training_instances)

        def predict(self, instances):
            # call base model's sklearner predict function
            return self.model.predict(instances)

        def set_params(self, params: Dict):
            # only the base model can be set in this simple example
            if params['model'] is not None:
                self.model = self.set_model(params['model'])
