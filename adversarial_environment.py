from typing import Dict
import factories
from adversaries.adversary import AdversaryStrategy
from data_reader import input, output
from learners.learner import InitialPredictor, ImprovedPredictor
from learners.models import sklearner
from learners.models.model import BaseModel

'''Methods to allow Adversarial environment to load instances from files.

  Args:
    data (str): Name of the dataset used to create instances.

  Returns:
    instances of type List[Instance]

'''
def set_train_instances(data):
  return input.load_instances([data, 'train'])


def set_test_instances(data):
  return input.load_instances([data, 'test'])


'''Collection of static methods for finding and instantiating key objects.

Provides the interface between adversarial environment and factory methods.

'''
def set_model(model_alg) -> BaseModel:
  if not type(model_alg) is str:
    return sklearner.Model(model_alg)
  else:
    factory = factories.ModelFactory(model_alg)
    factory.load_class()
    return factory.get_class()


def set_learn_initial(learner_model_alg) -> InitialPredictor:
  factory = factories.InitialPredictorFactory(learner_model_alg)
  factory.load_class()
  return factory.get_class()


def set_adversary(adversary_alg) -> AdversaryStrategy:
  if adversary_alg is None:
    return None
  else:
    factory = factories.AdversaryFactory(adversary_alg)
    factory.load_class()
    return factory.get_class()


def set_learner_improve(learner_improve_alg) -> ImprovedPredictor:
  if learner_improve_alg is None:
    return None
  else:
    factory = factories.ImprovedPredictorFactory(learner_improve_alg)
    factory.load_class()
    return factory.get_class()


class Battle(object):
  """Lower level implementation for a given battle.

  Contains a single learner, adversary, and improved predictor
  (implementations specified by the user). High-level proceedings
  of generalized adversarial stages:
    1. learner train
    2. adversary transform
    3. learner improve

    """

  def __init__(self, learner, adversary, improved_learner):
    """Create a new battle, in which no actions have been taken.

    Args:
        learner (InitialPredictor): Initial learning capabilities.
        adversary (AdversaryStrategy): Adversarial response.
        improved_learner (ImprovedPredictor): Improved learning methods.

    """
    self.learner = learner                    # type: InitialPredictor
    self.adversary = adversary                # type: AdversaryStrategy
    self.improved_learner = improved_learner  # type: ImprovedPredictor

    #: Generalized stage awareness for error state handling.
    self.stage_completed = {'1': False,
                            '2': False,
                            '3': False }

  def get_learner(self) -> InitialPredictor:
    return self.learner

  def get_adversary(self) -> AdversaryStrategy:
    return self.adversary

  def get_improved_learner(self) -> ImprovedPredictor:
    return self.improved_learner

  def stage_1(self, instances):
    """Learner initial training.

    Args:
        instances (List[Instance]): Train instances.

    """
    self.learner.train(instances)
    self.stage_completed['1'] = True

  def stage_2(self, train_instances, test_instances):
    """Adversary reaction.

    Gives the adversary full information about the initial learner
    and initial training instances. Allows adversary to choose
    relevant information and transform provided instances.

    Notes:
        Initial learning must take place first in order for this to work.

    Args:
        train_instances (List[Instance]): Train instances.
        test_instances (List[Instance]): Test instances.

    Returns:
        Transformed instances (optimal adversary response).

    """
    if not self.stage_completed['1']:
      raise AttributeError('Must perform initial training before adversarial reaction.')

    else:
      self.adversary.set_adversarial_params(self.learner, train_instances)
      adversary_instances = self.adversary.change_instances(test_instances)

      self.stage_completed['2'] = True
      return adversary_instances

  def stage_3(self, train_instances):
    """Learner improvement.

    Gives the improved learner full information about the initial learner
    and adversary. Allows improved learner to choose relevant information
    in order to improve learning model.

    Notes:
        Initial learning and adversarial reactionmust take place first in
        order for this to work.

    Args:
        train_instances (List[Instance]): Train instances.

    """
    if not self.stage_completed['1']:
      raise AttributeError('Must perform initial training before making improvements.')

    elif not self.stage_completed['2']:
      raise AttributeError('Improved learner does not have adversary to react to.')

    else:
      self.improved_learner.set_adversarial_params(self.learner, self.adversary)
      self.improved_learner.improve(train_instances)
      self.stage_completed['3'] = True


class Environment(object):
  """API for library usage.
  
  User specifies the underlying dataset, name of the environment (for persistence
  between executions), whether or not to save the environment state,
  learning model, and learner and adversary strategies.
  
  This is the intended interface for all library usage (except for the data extractor/
  initial instance generation). Point of entry for debugging & understanding library
  structure.
  
    """
  LEARNER = 'learner'
  ADVERSARY = 'adversary'
  IMPROVED_LEARNER = 'improved_learner'
  
  def __init__(self, data):
    """Create a new environment.
  
    Notes:
        Environment is built around data. Requires existence of the files containing
        instances in both train and test categories.
  
    Args:
        data (str): Name of the dataset to use throughout battle.
  
    """
    self.data = data
    self.train_instances = set_train_instances(data)
    self.test_instances = set_test_instances(data)
    self.transformed_instances = None
    self.predictions = None
  
    #: Each environment has exactly one battle at any given time.
    self.battle = None                                    # type: Battle
  
    #: Default values; overridden during environment creation.
    self.save_state = False
    self.name = None
  
  def create(self, adversary_alg, model_alg, learn_alg='learner',
             learner_improve_alg='learner', save_state=False, name='default'):
    """Create a battle for the environment.
  
    Args:
        adversary_alg (str): Name of adversary strategy to use (module name).
        model_alg (str or object): Learner model. Can specify a library-defined
        model or an sklearn classifier.
        learn_alg (str): Name of initial learn strategy to use (module name).
            learner_improve_alg (str): Name of improved learn strategy to use (module name).
            save_state (bool): Whether or not to save battle between executions.
            name (str): Name of the environment. Unique value for saving state, transformed
            instances, and preditions.
  
    """
    if adversary_alg is 'stackelberg' or adversary_alg is 'nash_equilibrium':
      learn_alg = adversary_alg
      learner_improve_alg = adversary_alg
  
    
    model = set_model(model_alg)
    learner = set_learn_initial(learn_alg)
    learner.set_model(model)
    adversary = set_adversary(adversary_alg)
    improved_learner = set_learner_improve(learner_improve_alg)
  
    self.battle = Battle(learner, adversary, improved_learner)
  
    self.save_state = save_state
    self.name = name
  
  def load_environment(self, name, save_state=False):
    """Load a battle for this environment.
  
    Args:
            save_state (bool): Whether or not to save battle between executions.
            name (str): Name of the environment.
  
    """
    self.battle = input.open_battle(name)
  
    self.save_state = save_state
    self.name = name
  
  def get_available_params(self, learner_type) -> Dict:
    """Return params and default values for specified type.
  
    Args:
        learner_type (str): learner, adversary, or improved learner.
  
    """
    if learner_type == self.LEARNER:
      params = self.battle.get_learner().get_available_params()
  
    elif learner_type == self.ADVERSARY:
      params = self.battle.get_adversary().get_available_params()
  
    elif learner_type == self.IMPROVED_LEARNER:
      params = self.battle.get_improved_learner().get_available_params()
  
    else:
      return None
  
    return params
  
  def set_params(self, learner_type, params: Dict):
    """Set params for specified type.
  
    Args:
        learner_type (str): learner, adversary, or improved learner.
        params (Dict): New param mappings.
  
    """
    if learner_type == self.LEARNER:
      self.battle.get_learner().set_params(params)
  
    elif learner_type == self.ADVERSARY:
      self.battle.get_adversary().set_params(params)
  
    elif learner_type == self.IMPROVED_LEARNER:
      self.battle.get_improved_learner().set_params(params)
  
    else:
      return None
  
  def train_learner(self):
    """Give train instances to battle and tell it to train its learner object.
  
    """
    self.battle.stage_1(self.train_instances)
  
    if self.save_state:
      output.save_battle(self.battle, self.name)
  
  def adversary_react(self, output_transformed_data=False):
    """Give train & test instances to battle and trigger adversary reaction.
  
    Args:
        output_transformed_data (bool): whether or not to save new instances
        to filesystem.
  
    """
    import pdb; pdb.set_trace()
    adversary_instances = self.battle.stage_2(self.train_instances, self.test_instances)
    self.transformed_instances = adversary_instances
  
    if output_transformed_data:
      output.save_transformed_instances(self.name, self.data, adversary_instances)
  
    if self.save_state:
      output.save_battle(self.battle, self.name)
  
  def improve_learner(self):
    """Give train instances to battle and trigger learner improve strategy.
  
    Args:
        output_transformed_data (bool): whether or not to save new instances
        to filesystem.
  
    """
    self.battle.stage_3(self.train_instances)
  
    if self.save_state:
      output.save_battle(self.battle, self.name)
  
  def test_learner(self, learner_type, output_predictions=False):
    """Now that the battle's over, check how the learner did.
  
    Args:
        learner_type (str): Initial learner or improved learner. Useful for
        accuracy comparisons.
        output_predictions (bool): whether or not to save predictions
        to filesystem.
  
    """
    predictions = None
    if learner_type == self.LEARNER:
      predictions = self.battle.get_learner().predict(self.test_instances)
  
    elif learner_type == self.IMPROVED_LEARNER:
      if self.transformed_instances is None:
        self.transformed_instances = input.load_instances([self.data+self.name, 'transformed'])
  
      predictions = self.battle.get_improved_learner().predict(self.transformed_instances)
  
    if output_predictions:
      output.save_predictions(self.name, self.data, predictions)
