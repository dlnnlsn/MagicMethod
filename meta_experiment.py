from experiment import *

def piecewise_linear_prediction_function(xs, ys):
    def f(x):
        for x1, x2, y1, y2 in zip([0] + list(xs), list(xs) + [1], [0] + list(ys), list(ys) + [1]):
            if x <= x2:
                return y1 + (y2 - y1) / (x2 - x1) * (x - x1)
        return 1
    return lambda s, d: f(s * d)

def piecewise_objective_function(actual_scores, test_weights, num_points, xs_and_ys):
    return minimize(piecewise_linear_prediction_function(xs_and_ys[:num_points], xs_and_ys[num_points:]), actual_scores, test_weights).fun

def piecewise_initial_data(num_points):
    equispaced = [(k + 1)/(num_points + 1) for k in range(num_points)]
    return equispaced + equispaced

def piecewise_bounds(num_points):
    return [(0, 1)] * (2 * num_points)

def piecewise_constraints(num_points):
    return [{"type": "ineq", "fun": lambda xs_and_ys: xs_and_ys[0]},
    *(
        {"type": "ineq", "fun": lambda xs_and_ys: xs_and_ys[k] - xs_and_ys[k - 1]}
        for k in range(1, num_points)
    ),
    {"type": "ineq", "fun": lambda xs_and_ys: 1 - xs_and_ys[num_points - 1]}]

def optimize_piecewise_prediction_function(num_points, actual_scores, test_weights):
    return scipy.optimize.minimize(partial(piecewise_objective_function, actual_scores, test_weights, num_points),
    piecewise_initial_data(num_points),
    bounds=piecewise_bounds(num_points),
    constraints=piecewise_constraints(num_points))

def exponential_prediction_function(coefficients):
    return lambda s, d: 1 - sum(c * np.exp(-(k + 1) * s * d) for (k, c) in enumerate(coefficients))

def exponential_initial_data(num_coefficients):
    return [1/num_coefficients] * num_coefficients

def exponential_contstraint(coefficients):
    return sum(coefficients) - 1

def exponential_objective_function(actual_scores, test_weights, coefficients):
    return minimize(exponential_prediction_function(coefficients), actual_scores, test_weights).fun

def optimize_exponential_prediction_function(num_coefficients, actual_scores, test_weights):
    return scipy.optimize.minimize(partial(exponential_objective_function, actual_scores, test_weights),
    exponential_initial_data(num_coefficients),
    bounds = [(0, 1)] * num_coefficients,
    constraints=[{"type": "eq", "fun": exponential_contstraint}])