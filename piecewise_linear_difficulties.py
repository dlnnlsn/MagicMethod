import numpy as np
import scipy
import scipy.optimize
from itertools import chain, starmap


def difficulty_fn(ms, bs):
    def d(x):
        output = 0
        bsum = 0
        for (m, b) in zip(ms, bs):
            if x <= bsum + b:
                return output + (x - bsum) * m
            bsum += b
            output += m * b
        return 1

    return d


def badness_function(actual_scores, test_weights, pts_per_difficulty_fn):
    students = len(actual_scores)
    tests = len(test_weights)

    def badness(all_params):
        ideal_scores = all_params[:students]
        ms = [
            all_params[students + k * pts_per_difficulty_fn * 2:students +
                       (2 * k + 1) * pts_per_difficulty_fn]
            for k in range(tests)
        ]
        bs = [
            all_params[students +
                       (2 * k + 1) * pts_per_difficulty_fn:students +
                       (2 * k + 2) * pts_per_difficulty_fn]
            for k in range(tests)
        ]
        ds = list(starmap(difficulty_fn, zip(ms, bs)))
        total = 0
        for i in range(students):
            for j in range(tests):
                if actual_scores[i][j] is not None and not np.isnan(
                        actual_scores[i][j]):
                    total += (
                        test_weights[j] *
                        (ds[j](ideal_scores[i]) - actual_scores[i][j])**2)
        return total

    return badness


def xcoord_constraint_fn(students, test_number, pts_per_difficulty_fn):
    def constraint(all_params):
        return (sum(
            all_params[students +
                       (2 * test_number + 1) * pts_per_difficulty_fn:students +
                       (2 * test_number + 2) * pts_per_difficulty_fn]) - 1)

    return constraint


def ycoord_constraint_fn(students, test_number, pts_per_difficulty_fn):
    def constraint(all_params):
        ms = all_params[students +
                        2 * test_number * pts_per_difficulty_fn:students +
                        (2 * test_number + 1) * pts_per_difficulty_fn]
        bs = all_params[students + (2 * test_number + 1) *
                        pts_per_difficulty_fn:students +
                        (2 * test_number + 2) * pts_per_difficulty_fn]
        return np.array(ms).dot(np.array(bs)) - 1

    return constraint


def constraints(students, tests, pts_per_difficulty_fn):
    return [
        *({
            "type": "eq",
            "fun": xcoord_constraint_fn(students, k, pts_per_difficulty_fn),
        } for k in range(tests)),
        *({
            "type": "eq",
            "fun": ycoord_constraint_fn(students, k, pts_per_difficulty_fn),
        } for k in range(tests)),
    ]


def bounds(students, tests, pts_per_difficulty_fn):
    return [(0, 1)] * students + list(
        chain(*([[(0, None)] * pts_per_difficulty_fn +
                 [(0, 1)] * pts_per_difficulty_fn] * tests)))


def initial_values_per_test(pts_per_difficulty_fn):
    return [1] * pts_per_difficulty_fn + [1 / pts_per_difficulty_fn
                                          ] * pts_per_difficulty_fn


def initial_value(students, tests, pts_per_difficulty_fn):
    return [0.5] * students + list(
        chain(*([initial_values_per_test(pts_per_difficulty_fn)] * tests)))


def minimize(actual_scores, test_weights, pts_per_difficulty_fn, **kwargs):
    students = len(actual_scores)
    tests = len(test_weights)
    return scipy.optimize.minimize(
        badness_function(actual_scores, test_weights, pts_per_difficulty_fn),
        initial_value(students, tests, pts_per_difficulty_fn),
        bounds=bounds(students, tests, pts_per_difficulty_fn),
        constraints=constraints(students, tests, pts_per_difficulty_fn),
        **kwargs)
