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

    def badness(all_params):
        scores = all_params[:students]
        mss = get_mss(students, tests, pts_per_difficulty_fn, all_params)
        dxss = get_dxss(students, tests, pts_per_difficulty_fn, all_params)
        total = 0
        for ms, dxs, w, a in zip(mss, dxss, test_weights, actual_scores.transpose()):
            d = difficulty_function(ms, dxs)
            total += np.nansum(w * (d(scores) - a)**2)
        return total

    return badness


def get_dxss(students, tests, pts_per_difficulty_fn, all_params):
    return all_params[students + tests *
                      pts_per_difficulty_fn:].reshape(tests, pts_per_difficulty_fn)


def get_mss(students, tests, pts_per_difficulty_fn, all_params):
    return all_params[students: students +
                      tests * pts_per_difficulty_fn].reshape(tests, pts_per_difficulty_fn)


def xcoord_constraint_fn(students, tests, pts_per_difficulty_fn):
    def constraint(all_params):
        dxss = get_dxss(students, tests, pts_per_difficulty_fn, all_params)
        return np.all(np.sum(dxss, 1) == 1) - 1

    return constraint


def ycoord_constraint_fn(students, tests, pts_per_difficulty_fn):
    def constraint(all_params):
        mss = get_mss(students, tests, pts_per_difficulty_fn, all_params)
        dxss = get_dxss(students, tests, pts_per_difficulty_fn, all_params)
        return np.all(np.sum(mss * dxss, 1) == 1) - 1

    return constraint


def constraints(students, tests, pts_per_difficulty_fn):
    return [
        {
            "type": "eq",
            "fun": xcoord_constraint_fn(students, tests, pts_per_difficulty_fn),
        },
        {
            "type": "eq",
            "fun": ycoord_constraint_fn(students, tests, pts_per_difficulty_fn),
        }
    ]


def bounds(students, tests, pts_per_difficulty_fn):
    return [(0, 1)] * students + [(0, None)] * (pts_per_difficulty_fn * tests) + [(0, 1)] * (pts_per_difficulty_fn * tests)


def initial_value(students, tests, pts_per_difficulty_fn):
    return np.append(np.ones(students) / 2, np.ones(pts_per_difficulty_fn * tests), np.ones(pts_per_difficulty_fn * tests) / pts_per_difficulty_fn)


def minimize(actual_scores, test_weights, pts_per_difficulty_fn, **kwargs):
    students = len(actual_scores)
    tests = len(test_weights)
    return scipy.optimize.minimize(
        badness_function(actual_scores, test_weights, pts_per_difficulty_fn),
        initial_value(students, tests, pts_per_difficulty_fn),
        bounds=bounds(students, tests, pts_per_difficulty_fn),
        constraints=constraints(students, tests, pts_per_difficulty_fn),
        **kwargs)
