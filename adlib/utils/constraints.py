# -------------------------------------------------------------------------------
# adversariaLib - Advanced library for the evaluation of machine
# learning algorithms and classifiers against adversarial attacks.
#
# Copyright (C) 2013, Igino Corona, Battista Biggio, Davide Maiorca,
# Dept. of Electrical and Electronic Engineering, University of Cagliari, Italy.
#
# adversariaLib is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# adversariaLib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------------
from numpy import array, absolute, zeros


def within_no_constraints(root_pattern, pattern, bound_no):
    return True


def apply_no_constraints(root_pattern, pattern, grad_update, grad_step, bound_no):
    return array(pattern - (grad_step * grad_update))


def within_hypercube(root_pattern, pattern, bound_no, box_step):
    bound = bound_no * box_step
    for i in range(len(pattern)):
        if (pattern[i] - root_pattern[i]) > bound or (pattern[i] - root_pattern[i]) < -bound:
            return False
    return True


def apply_hypercube(root_pattern, pattern, grad_update, grad_step, bound_no, box_step):
    next_pattern = apply_no_constraints(root_pattern, pattern, grad_update, grad_step, bound_no)
    bound = bound_no * box_step
    for i in range(len(pattern)):
        if (next_pattern[i] - root_pattern[i]) > bound:
            next_pattern[i] = root_pattern[i] + bound
        elif (next_pattern[i] - root_pattern[i]) < -bound:
            next_pattern[i] = root_pattern[i] - bound
    return next_pattern


def within_only_increment(root_pattern, pattern, bound_no, only_increment_step, weights,
                          inv_weights, feature_upper_bound):
    root_patt_discrete = array([int(round(item)) for item in root_pattern * weights])
    patt_discrete = array([int(round(item)) for item in pattern * weights])
    # we can avoid checking whether pattern < root_pattern,
    # since it is assumed that pattern is always >= root_pattern
    dist = sum(absolute(patt_discrete - root_patt_discrete))
    if dist <= only_increment_step * bound_no:
        return True
    return False


def apply_only_increment(root_pattern, pattern, grad_update, grad_step, bound_no,
                         only_increment_step, weights, inv_weights, feature_upper_bound=None):
    grad_update_with_indexes = list(enumerate(grad_update * inv_weights))
    # ATTENZIONE. IL GRADIENTE VA MOLT PER I PESI/STEP SULLE SINGOLE FEATURE
    grad_update_with_indexes.sort(lambda x, y: -cmp(abs(x[1]), abs(y[1])))
    new_pattern = array(pattern)
    for feature_index, value in grad_update_with_indexes:
        if value > 0:  # it means that we should DECREMENT
            new_value = pattern[feature_index] - inv_weights[feature_index]
            if root_pattern[feature_index] <= new_value:
                # we are under "only increment" and normalization is OK...
                new_pattern[feature_index] = new_value
                return new_pattern
            # we go to the next candidate feature
            continue
        elif value < 0:  # it means that we should INCREMENT
            diff_pattern = [int(round(item)) for item in absolute(pattern - root_pattern) * weights]
            new_value = new_pattern[feature_index] + inv_weights[feature_index]
            if (sum(diff_pattern) < only_increment_step * bound_no) and (
                    feature_upper_bound is None or (
                    new_value <= feature_upper_bound)):
                # we are under the boundary and upper_bound is OK...
                new_pattern[feature_index] = new_value  # we increment the feature
                return new_pattern
            # we go to the next candidate feature
            continue
        else:
            return new_pattern
    return new_pattern


def within_hamming(root_pattern, pattern, bound_no):
    """For simplicity, we DO NOT check if features are TRULY binary..."""
    if sum(absolute(pattern - root_pattern)) <= bound_no:
        return True
    else:
        return False


def apply_hamming(root_pattern, pattern, grad_update, grad_step, bound_no):
    """For simplicity, we DO NOT check if features are TRULY binary..."""
    grad_update_with_indexes = list(enumerate(grad_update))
    grad_update_with_indexes.sort(lambda x, y: -cmp(abs(x[1]), abs(y[1])))
    new_pattern = array(pattern)
    for feature_index, value in grad_update_with_indexes:
        if value > 0:  # it means that we should DECREMENT
            if pattern[feature_index] == 1:
                new_pattern[feature_index] = 0
                return new_pattern
            else:
                # we go to the next candidate feature
                continue
        elif value < 0:  # it means that we should INCREMENT
            if sum(absolute(pattern - root_pattern)) < bound_no and new_pattern[feature_index] == 0:
                # we are under the boundary
                new_pattern[feature_index] = 1  # we increment the feature
                return new_pattern
            continue
        else:
            return new_pattern
    return new_pattern


def within_box_fixed(root_pattern, pattern, bound_no, lb, ub):
    for i in range(len(pattern)):
        if (pattern[i]) > ub or (pattern[i]) < lb:
            return False
    return True


def apply_box_fixed(root_pattern, pattern, grad_update, grad_step, bound_no, lb, ub):
    next_pattern = apply_no_constraints(root_pattern, pattern, grad_update, grad_step, bound_no)
    for i in range(len(pattern)):
        if next_pattern[i] > ub:
            next_pattern[i] = ub
        elif next_pattern[i] < lb:
            next_pattern[i] = lb
    return next_pattern
