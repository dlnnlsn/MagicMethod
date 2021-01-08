import numpy as np


def badness(difficulties, scores, test_weights, actual_scores):
    return np.nansum(
        test_weights * (scores[np.newaxis].T * difficulties - actual_scores) ** 2
    )


def scores_from_difficulties(difficulties, test_weights, actual_scores, score_mask):
    numerators = np.nansum(difficulties * test_weights * actual_scores, axis=1)
    denominators = np.nansum(test_weights * difficulties ** 2 * score_mask, axis=1)
    return numerators / denominators


def difficulties_from_scores(scores, test_weights, actual_scores, score_mask):
    denominators = np.nansum(
        scores[np.newaxis].T ** 2 * score_mask * test_weights, axis=0
    )
    numerators = np.nansum(scores[np.newaxis].T * actual_scores * test_weights, axis=0)
    lambda_coeff = np.sum(1 / denominators)
    lambda_value = (1 - numerators.dot(denominators)) / lambda_coeff
    return (lambda_value + numerators) / denominators


def calculate_scores_and_difficulties(test_weights, actual_scores):
    pass
