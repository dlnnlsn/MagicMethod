import numpy as np


def badness(difficulties, scores, test_weights, actual_scores):
    return np.nansum(
        test_weights * (scores[np.newaxis].T * difficulties - actual_scores) ** 2
    )


def scores_from_difficulties(difficulties, test_weights, actual_scores):
    pass


def difficulties_from_scores(scores, test_weights, actual_scores):
    pass


def calculate_scores_and_difficulties(test_weights, actual_scores):
    pass
