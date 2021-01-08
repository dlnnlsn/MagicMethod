import numpy as np


def badness(difficulties, scores, test_weights, actual_scores):
    return np.nansum(
        test_weights * (scores[np.newaxis].T * difficulties - actual_scores) ** 2
    )


def scores_from_difficulties(difficulties, test_weights, actual_scores):
    pass


def difficulties_from_scores(scores, test_weights, actual_scores):
    denominators = np.nansum(actual_scores ** 2 * test_weights, axis=0)
    numerators = np.nansum(scores[np.newaxis].T * actual_scores * test_weights, axis=1)
    lambda_coeff = np.sum(1 / denominators)
    lambda_value = (1 - numerators.dot(denominators)) / lambda_coeff
    return (lambda_value + numerators) / denominators


def calculate_scores_and_difficulties(test_weights, actual_scores):
    pass
