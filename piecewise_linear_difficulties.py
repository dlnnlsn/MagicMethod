import numpy as np
import scipy
import scipy.optimize
from itertools import chain, starmap


def difficulty_function(ms, dxs):
    ys = ms * dxs
    ys = ys.cumsum() - ys
    xs = dxs.cumsum() - dxs

    def d(s):
        index = np.searchsorted(xs, s) - 1
        return (s > 0) * (ys[index] + ms[index] * (s - xs[index]))
    return d


def badness_function(actual_scores, test_weights, pts_per_difficulty_fn):
    students = len(actual_scores)
    tests = len(test_weights)

    def badness(allparams):
        scores = allparams[:students]
        mss = allparams[students: students +
                        tests * pts_per_difficulty_fn].reshape(tests, pts_per_difficulty_fn)
        dxss = allparams[students + tests *
                        pts_per_difficulty_fn:].reshape(tests, pts_per_difficulty_fn)
        total = 0
        for ms, dxs, w, a in zip(mss, dxss, test_weights, actual_scores):
            d = difficulty_function(ms, dxs)
            total += np.nansum(w * (d(scores) - a)**2)
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
        dxs = all_params[students + (2 * test_number + 1) *
                        pts_per_difficulty_fn:students +
                        (2 * test_number + 2) * pts_per_difficulty_fn]
        return np.array(ms).dot(np.array(dxs)) - 1

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
