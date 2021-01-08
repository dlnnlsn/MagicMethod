import numpy as np


def badness(difficulties, scores, test_weights, actual_scores):
    return np.nansum(
        test_weights * (scores[np.newaxis].T * difficulties - actual_scores) ** 2
    )


def scores_from_difficulties(
    difficulties, weight_matrix, actual_scores, weight_matrix_times_actual_scores
):
    numerators = np.sum(difficulties * weight_matrix_times_actual_scores, axis=1)
    denominators = np.sum(weight_matrix * difficulties ** 2, axis=1)
    return numerators / denominators


def difficulties_from_scores(
    scores, weight_matrix, actual_scores, weight_matrix_times_actual_scores
):
    denominators = np.sum(scores[np.newaxis].T ** 2 * weight_matrix, axis=0)
    numerators = np.sum(
        scores[np.newaxis].T * weight_matrix_times_actual_scores, axis=0
    )
    lambda_coeff = np.sum(1 / denominators)
    lambda_value = (1 - numerators.dot(denominators)) / lambda_coeff
    return (lambda_value + numerators) / denominators


def calculate_scores_and_difficulties(
    test_weights, actual_scores, tolerance=1e-3, max_iterations=100
):
    weight_matrix = ~np.isnan(actual_scores) * test_weights
    weight_matrix_times_actual_scores = weight_matrix * np.nan_to_num(actual_scores)
    _, tests = actual_scores.shape
    difficulties = np.ones(tests) / tests
    scores = scores_from_difficulties(
        difficulties, weight_matrix, actual_scores, weight_matrix_times_actual_scores
    )
    previous_badness = badness(difficulties, scores, test_weights, actual_scores)
    for _ in range(max_iterations):
        difficulties = difficulties_from_scores(
            scores, weight_matrix, actual_scores, weight_matrix_times_actual_scores
        )
        scores = scores_from_difficulties(
            difficulties,
            weight_matrix,
            actual_scores,
            weight_matrix_times_actual_scores,
        )
        new_badness = badness(difficulties, scores, test_weights, actual_scores)
        if abs(new_badness - previous_badness) < tolerance:
            return difficulties, scores
    return difficulties, scores
