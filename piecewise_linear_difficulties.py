import numpy as np
import scipy
import scipy.optimize
from functools import partial
from itertools import chain, starmap


def difficulty_function(ms, bs):
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


def badness_function(actual_scores, test_weights, points_per_difficulty_function):
    students = len(actual_scores)
    tests = len(test_weights)

    def badness(all_params):
        ideal_scores = all_params[:students]
        ms = [
            all_params[
                students
                + k * points_per_difficulty_function * 2 : students
                + (2 * k + 1) * points_per_difficulty_function
            ]
            for k in range(tests)
        ]
        bs = [
            all_params[
                students
                + (2 * k + 1) * points_per_difficulty_function : students
                + (2 * k + 2) * points_per_difficulty_function
            ]
            for k in range(tests)
        ]
        ds = list(starmap(difficulty_function, zip(ms, bs)))
        total = 0
        for i in range(students):
            for j in range(tests):
                if actual_scores[i][j] is not None and not np.isnan(
                    actual_scores[i][j]
                ):
                    total += (
                        test_weights[j]
                        * (ds[j](ideal_scores[i]) - actual_scores[i][j]) ** 2
                    )
        return total

    return badness


def xcoordinate_constraint_function(
    students, test_number, points_per_difficulty_function
):
    def constraint(all_params):
        return (
            sum(
                all_params[
                    students
                    + (2 * test_number + 1) * points_per_difficulty_function : students
                    + (2 * test_number + 2) * points_per_difficulty_function
                ]
            )
            - 1
        )

    return constraint


def ycoordinate_constraint_function(
    students, test_number, points_per_difficulty_function
):
    def constraint(all_params):
        ms = all_params[
            students
            + 2 * test_number * points_per_difficulty_function : students
            + (2 * test_number + 1) * points_per_difficulty_function
        ]
        bs = all_params[
            students
            + (2 * test_number + 1) * points_per_difficulty_function : students
            + (2 * test_number + 2) * points_per_difficulty_function
        ]
        return np.array(ms).dot(np.array(bs)) - 1

    return constraint


def constraints(students, tests, points_per_difficulty_function):
    return [
        *(
            {
                "type": "eq",
                "fun": xcoordinate_constraint_function(
                    students, k, points_per_difficulty_function
                ),
            }
            for k in range(tests)
        ),
        *(
            {
                "type": "eq",
                "fun": ycoordinate_constraint_function(
                    students, k, points_per_difficulty_function
                ),
            }
            for k in range(tests)
        ),
    ]


def bounds(students, tests, points_per_difficulty_function):
    return [(0, None)] * students + list(
        chain(
            *(
                [
                    [(0, None)] * points_per_difficulty_function
                    + [(0, 1)] * points_per_difficulty_function
                ]
                * tests
            )
        )
    )


def initial_values_per_test(points_per_difficulty_function):
    return [1] * points_per_difficulty_function + [
        1 / points_per_difficulty_function
    ] * points_per_difficulty_function


def initial_value(students, tests, points_per_difficulty_function):
    return [0.5] * students + list(
        chain(*([initial_values_per_test(points_per_difficulty_function)] * tests))
    )


def minimize(actual_scores, test_weights, points_per_difficulty_function, **kwargs):
    students = len(actual_scores)
    tests = len(test_weights)
    return scipy.optimize.minimize(
        badness_function(actual_scores, test_weights, points_per_difficulty_function),
        initial_value(students, tests, points_per_difficulty_function),
        bounds=bounds(students, tests, points_per_difficulty_function),
        constraints=constraints(students, tests, points_per_difficulty_function),
        **kwargs
    )
