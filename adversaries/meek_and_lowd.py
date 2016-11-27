from typing import List, Dict
from adversaries.adversary import AdversaryStrategy
from data_reader.input import Instance, FeatureVector
from learners.learner import InitialPredictor
from data_reader import operations
from copy import deepcopy
from itertools import filterfalse
from math import sqrt

'''Meek and Lowd IMAC.

Concept:
    Given an ideal adversarial instance A and an instance classified as
    non adversarial X, finds the feature set difference between A and X
    and iteratively applies those features to A, rolling each change back
    if it results in an adversarial classification under the learner's model.
'''

class ReverseEngineerClassifier:

    def __init__(self, adversary):
        self.adversary = adversary
    
    def execute(self):
        # Document
        raise NotImplementedError

class FindBooleanIMAC(ReverseEngineerClassifier):

    def execute(self, instance):
        x_minus = self.adversary.negative_instance.get_feature_vector()
        y = deepcopy(x_minus)
        # assume that instance is the target instance
        xa = instance.get_feature_vector()
        while True:
            y_prev = deepcopy(y)
            C_y = y.feature_difference(xa)

            for index in C_y:
                y.flip_bit(index)
                if self.adversary.learn_model.predict(Instance(0,y)) == 1:
                    y.flip_bit(index)

            C_y = y.feature_difference(xa)
            not_C_y = list(filterfalse(lambda x: x in C_y, range(0, y.feature_count)))
            # in testing, this iterates through over 2 mill combinations and is very slow 
            for index1 in C_y:
                for index2 in C_y:
                    if index2 <= index1: continue
                    for index3 in not_C_y:
                        y.flip_bit(index1)
                        y.flip_bit(index2)
                        y.flip_bit(index3)
                        if self.adversary.learn_model.predict(Instance(0,y)) == 1:
                            y.flip_bit(index1)
                            y.flip_bit(index2)
                            y.flip_bit(index3)

            if operations.fv_equals(y, y_prev):
                return Instance(1, y)

class MultiLineSearch(ReverseEngineerClassifier):

    def execute(self, instance):
        search_set, cost_max = self.convex_set_search(
            self.adversary.cost_vector,
            instance
        )
        # use spiral search to generate narrower bounds and prune search set
        #search_set, cost_min, cost_max = self.spiral_search(search_set, cost_max, instance)
        cost_min = 1
        return self.multi_line_search(search_set, cost_min, cost_max, self.adversary.epsilon, instance)

    # W: list of search vectors of unit cost
    # C_plus = initial lower bound on cost
    # C_minus = initial upper bound on cost
    # x_a is the ideal minimum cost instance
    def multi_line_search(self, W, cost_plus, cost_minus, epsilon, instance):
        x_minus = self.adversary.negative_instance.get_feature_vector()
        x_star = deepcopy(x_minus)
        # x_a subset of positive instances
        x_a = instance.get_feature_vector()
        while cost_minus / cost_plus > 1 + epsilon:
            cost_t = sqrt(cost_plus * cost_minus)
            negative_vertex_found = False
            print('min cost', cost_plus, 'max cost', cost_minus, 'ratio: ', cost_minus / cost_plus)
            for e in W:
                # looks like we're assuming that y and e have the same dimensions
                # TODO: incorporate cost_t
                new_feature_vector = FeatureVector(x_a.feature_count, x_a.indices)
                new_feature_vector.flip_bit(e)
                query_result = self.adversary.learn_model.predict(Instance(0, new_feature_vector))
                print('changed feature: %s, classified as: %s' % (e, query_result))
                if query_result == InitialPredictor.negative_classification:
                    x_star = new_feature_vector
                    negative_vertex_found = True
                    # Prune all costs that result in positive prediction
                    new_W = set(W)
                    for i in W:
                        tmp_feature_vector = FeatureVector(x_star.feature_count, x_star.indices)
                        tmp_feature_vector.flip_bit(i)
                        if self.adversary.learn_model.predict(Instance(0, tmp_feature_vector)) == 1:
                            # deleting from a list when you iterate through it may cause bugs
                            new_W.remove(i)
                    break
            cost_next_plus = cost_plus
            cost_next_minus = cost_minus
            W = new_W
        if negative_vertex_found:
            cost_next_minus = cost_t
        else:
            cost_next_plus = cost_t
        return Instance(1, x_star)
    
    # I think the algorithm expects you to pass in a cost vector for each feature
    # But in feature difference we assume a uniform adversarial cost function
    # I'm not sure what value epsilon is supposed to be (or cost_min)
    def convex_set_search(self, cost_vector, instance):
        x_a = instance.get_feature_vector()
        x_minus = self.adversary.negative_instance.get_feature_vector()
        dimension = len(x_a)
        cost_max = self.adversary.negative_instance.get_feature_vector_cost(x_a, cost_vector);
        search_set = set()
        for i in range(dimension):
            # TODO: x_a[i] is wrong, its actually [0,0,0,..,1,..,0] where 1 is the ith feature
            element = 1/(instance.get_feature_cost(cost_vector, i)) * x_a[i] # Not sure about this
            search_set.add(element)
            search_set.add(-element)
        return (search_set, cost_max)

    # search_set: set()
    # cost_max: int
    # Finds a lower bound on the MAC if one exists
    def spiral_search(self, search_set, cost_max, instance):
        x_a = instance.get_feature_vector()
        new_search_set = set()
        t = 0
        while len(search_set) > 0:
            element = search_set.pop()
            new_search_set.add(element)
            # TODO: include prediction weight in prediction!!!!
            prediction_weight = cost_max * pow(2, pow(-2,t))
            new_feature_vector = FeatureVector(x_a.feature_count, x_a.indices)
            new_feature_vector.flip_bit(element)
            query_result = self.adversary.learn_model.predict(
              Instance(0, new_feature_vector)
            )
            if query_result == InitialPredictor.negative_classification:
                search_set.add(element)
                new_search_set = set()
                t += 1
        cost_min = cost_max * pow(2, -pow(2, t))
        if t > 0: cost_max = cost_max * pow(2, -pow(2, t - 1))
        return (new_search_set, cost_min, cost_max)

class KStepMultiLineSearch(MultiLineSearch):

    def multi_line_search(self, search_set, cost_min, cost_max, epsilon, instance):
        x_a = instance.get_feature_vector()
        x_star = self.adversary.negative_instance.get_feature_vector()
        while cost_max / cost_min > 1 + epsilon:
            e = search_set.pop() # pop removes an element which might not be what we want to do
            temp_min_cost = cost_min
            temp_max_cost = cost_max
            for i in range(self.adversary.k):
                tmp_feature_vector = FeatureVector(x_a.feature_count, x_a.indices)
                tmp_feature_vector.flip_bit(e)
                temp_cost = sqrt(temp_min_cost * temp_max_cost)
                query_result = self.adversary.learn_model.predict(Instance(0, tmp_feature_vector))
                if query_result == 1:
                    temp_min_cost = temp_cost
                else:
                    temp_max_cost = temp_cost
                    x_star = x_a + temp_cost * e
            positive_directions = set()
            for i in search_set: # we know i != e because e was removed
                tmp_feature_vector = FeatureVector(x_a.feature_count, x_a.indices)
                tmp_feature_vector.flip_bit(i)
                query_result = self.adversary.learn_model.predict(Instance(0, tmp_feature_vector))
                if query_result == -1:
                    x_star = x_a + temp_min_cost * i
                    # Proon positive directions
                    for k in positive_directions:
                        search_set.remove(k)
                    break
                else:
                    positive_directions.add(i)
            cost_max = temp_max_cost
            if len(positive_directions) > 0:
                cost_min = temp_min_cost
        return Instance(1, x_star)

class Adversary(AdversaryStrategy):

    FIND_BOOLEAN_IMAC = 'find_boolean_IMAC'
    MULTI_LINE_SEARCH = 'multi_line_search'
    K_STEP_MULTI_LINE_SEARCH = 'k_step_multi_line_search'

    def __init__(self):
        self.learn_model = None                # type: InitialPredictor
        self.positive_instance = None    # type: Instance
        self.negative_instance = None    # type: Instance
        self.adversary_costs = None    # type: Array 
        self.epsilon = None    # type: Double?
        self.cost_min = None    # type: Double?
        self.k = None    # type: integer
        self.attack_model_type = Adversary.FIND_BOOLEAN_IMAC
        self.cost_vector = None

    def change_instances(self, instances: List[Instance]) -> List[Instance]:
        '''Change adversarial instances by finding boolean IMAC.

        used to transform all test instances, assuming that each instance classified as
        adversarial is the 'ideal' instance. To find a single IMAC based on a user specified
        adversarial 'ideal' instance, override test data to pass in single (or multiple)
        adversarially preferred instances.

        'ideal' instance refers to the test vector that makes the best argument for the
        adversaries intention. E.g. spam filtering, email that the adversary classifies as being
        most useful to their efforts
        '''
        transformed_instances = []

        for instance in instances:
            transformed_instance = deepcopy(instance)
            if instance.get_label() == 1:
                transformed_instances.append(self.attack_model.execute(instance))
            else:
                transformed_instances.append(transformed_instance)
        return transformed_instances

    def get_available_params(self):
        return {
            'attack_model_type': self.attack_model_type,
            'adversary_costs': self.adversary_costs,
            'epsilon': self.epsilon,
            'cost_min': self.cost_min,
            'k': self.k,
            'cost_vector': self.cost_vector,
        }

    def set_params(self, params: Dict):
        if params['attack_model_type'] is not None:
            self.attack_model_type = params['attack_model_type']
        
        if params['adversary_costs'] is not None:
            self.adversary_costs = params['adversary_costs']

        if params['epsilon'] is not None:
            self.epsilon = params['epsilon']
        
        if params['cost_min'] is not None:
            self.cost_min = params['cost_min']
        
        if params['k'] is not None:
            self.k = params['k']
        if params['cost_vector'] is not None:
            self.cost_vector = params['cost_vector']
        
        return None

    def set_adversarial_params(self, learner, train_instances):
        self.learn_model = learner
        instances = train_instances # type: List[Instance]
        self.positive_instance = next((x for x in instances if x.get_label() == 1), None)
        self.negative_instance = next((x for x in instances if x.get_label() == -1), None)

        if self.attack_model_type is Adversary.FIND_BOOLEAN_IMAC:
            self.attack_model = FindBooleanIMAC(self)

        elif self.attack_model_type is Adversary.MULTI_LINE_SEARCH:
            self.attack_model = MultiLineSearch(self)

        elif self.attack_model_type is Adversary.K_STEP_MULTI_LINE_SEARCH:
            self.attack_model = KStepMultiLineSearch(self)

        else:
          raise ValueError('invalid attack model type')

    # This is a uniform adversarial cost function, should we add a weight parameter?
    # TODO: figure out if I need to implement these
    # TODO: fix parameters
    def find_continuous_weights(self, instances):
        return None

    # TODO: fix parameters
    def find_continuous_IMAC(self, instances):
        return None
