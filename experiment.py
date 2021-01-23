import numpy as np
import scipy
import scipy.special
import scipy.optimize
from functools import partial

def objective(f,actual_scores,test_weights):
    students = len(actual_scores)
    tests = len(test_weights)
    def obj(scores_and_difficulties):
        scores = scores_and_difficulties[:students]
        difficulties = scores_and_difficulties[students:]
        total = 0
        for i in range(students):
            for j in range(tests):
                if actual_scores[i][j] is not None and not np.isnan(actual_scores[i][j]):
                    total += test_weights[j] * (f(scores[i],difficulties[j]) - actual_scores[i][j])**2
        return total
    return obj

def initial(students,tests):
    return np.array([1] * students + [1/tests] * tests)

def normalise_difficulties(students,scores_and_difficulties):
    return np.sum(scores_and_difficulties[students:]) - 1

def get_bounds(students,tests):
    return [(0,None)] * students + [(0,1)] * tests

def minimize(f,actual_scores,test_weights,**kwargs):
    students = len(actual_scores)
    tests = len(test_weights)
    x0 = initial(students,tests)
    bounds = get_bounds(students,tests)
    difficulty_constraint = partial(normalise_difficulties,students)
    constraints = [{"type": "eq","fun": difficulty_constraint}]
    return scipy.optimize.minimize(objective(f,actual_scores,test_weights),x0,bounds=bounds,constraints=constraints,**kwargs)

relu = lambda k: lambda s,d: 1 - np.log(1 + np.exp(-k * d * s)) / np.log(2)

candidate_functions = {
    "Logistic Function" : lambda s,d: 2/(1 + np.exp(-d * s)) - 1,
    "Error Function": lambda s,d: scipy.special.erf(d * s),
    "Hyperbolic Tangent": lambda s,d: np.tanh(d * s),
    "Gudermannian": lambda s,d: 4 * np.arctan(np.tanh(d * s / 2)) / np.pi,
    "Normal Product": lambda s,d: d * s,
    "Arctan": lambda s,d: 2 * np.arctan(d * s) / np.pi,
    "Cutoff Product": lambda s,d: min(d * s,1),
    "x/sqrt(1 + x^2)": lambda s,d: s * d / np.sqrt(1 + (d * s)**2),
    "x/(1 + x)": lambda s,d: s * d / (1 + s * d),
    "RELU(1)": relu(1),
    "RELU(2)": relu(2),
    "RELU(4)": relu(4),
    "RELU(8)": relu(8),
    "RELU(16)": relu(16),
    "Smoothstep 3": lambda s,d: (lambda x: 0 if x <= 0 else 3*x**2 - 2*x**3 if x <= 1 else 1)(s * d),
    "Smoothstep 5": lambda s,d: (lambda x: 0 if x <= 0 else 6*x**5 - 15*x**4 + 10*x**3 if x <= 1 else 1)(s * d),
    "Smoothstep 7": lambda s,d: (lambda x: 0 if x <= 0 else -20*x**7 + 70*x**6 - 84*x**5 + 35*x**4 if x <= 1 else 1)(s * d),
    "Smoothstep 9": lambda s,d: (lambda x: 0 if x <= 0 else 70*x**9 - 315*x**8 + 540*x**7 - 420*x**6 + 126*x**5 if x <= 1 else 1)(s * d),

}

def apply_all_candidate_functions(actual_scores,test_weights):
    return dict((description,minimize(function,actual_scores,test_weights).fun) for (description,function) in candidate_functions.items())

def random_trial(students,tests,probability_of_writing_test):
    actual_scores = np.random.uniform(0,1,(students,tests))
    weights = np.random.uniform(0,1,tests)
    not_written = np.random.uniform(0,1,(students,tests)) < (1 - probability_of_writing_test)
    actual_scores[not_written] = np.nan
    return apply_all_candidate_functions(actual_scores,weights)

raw_scores = """71.4,95.0,31.0,,,27.0,29.0,19.0,35.0,29.0
81.0,100.0,,,,8.0,24.0,15.0,26.0,19.0
64.3,81.0,24.0,,,14.0,32.0,21.0,14.0,25.0
47.6,89.0,,,,12.0,21.0,17.0,21.0,16.0
66.7,62.0,18.0,,,14.0,14.0,18.0,24.0,15.0
61.9,80.0,,,,13.0,18.0,18.0,12.0,10.0
33.3,74.0,16.0,,,12.0,21.0,7.0,23.0,11.0
35.7,71.0,19.0,,,10.0,18.0,19.0,18.0,10.0
57.1,74.0,11.0,,,3.0,15.0,14.0,13.0,10.0
54.8,69.0,28.0,,,12.0,13.0,9.0,10.0,12.0
47.6,78.0,19.0,,,4.0,13.0,6.0,9.0,12.0
54.8,65.0,13.0,,,8.0,11.0,8.0,9.0,9.0
54.8,72.0,,,,5.0,7.0,4.0,10.0,10.0
54.8,77.0,11.0,,,2.0,9.0,4.0,2.0,10.0
,96.0,,,,3.0,6.0,2.0,5.0,1.0
26.2,71.0,,,,4.0,6.0,1.0,8.0,4.0
40.5,68.0,25.0,,,1.0,0.0,5.0,3.0,1.0
,60.0,,,,4.0,7.0,1.0,7.0,2.0
40.5,71.0,,,,1.0,2.0,1.0,0.0,0.0
,,,100.0,34.0,15.0,7.0,14.0,14.0,11.0
21.4,75.0,,,,0.0,0.0,3.0,5.0,1.0
14.3,63.0,,,,1.0,6.0,0.0,5.0,2.0
,,,96.0,,2.0,12.0,12.0,7.0,10.0
,,,78.0,45.0,3.0,6.0,3.0,4.0,4.0
,,,95.0,,0.0,5.0,3.0,6.0,0.0
,,,87.0,,1.0,3.0,3.0,7.0,1.0
,,,91.0,15.0,0.0,0.0,3.0,7.0,2.0
,,,87.0,32.0,1.0,2.0,0.0,0.0,0.0
,,,87.0,23.0,0.0,4.0,0.0,4.0,0.0
,,,91.0,,0.0,0.0,0.0,4.0,0.0
,,,92.0,5.0,0.0,0.0,0.0,7.0,1.0
,,,96.0,5.0,0.0,1.0,1.0,2.0,1.0
,,,82.0,11.0,0.0,2.0,0.0,8.0,1.0
,,,70.0,12.0,2.0,6.0,1.0,8.0,1.0
,,,78.0,,1.0,0.0,3.0,4.0,1.0
,,,82.0,11.0,0.0,3.0,5.0,0.0,0.0
,,,74.0,5.0,1.0,5.0,0.0,7.0,0.0
,,,78.0,7.0,0.0,2.0,1.0,4.0,0.0
,,,65.0,,0.0,0.0,0.0,0.0,0.0"""

scores = raw_scores.split("\n")
scores = [[float(x) if x else np.nan for x in line.split(",")] for line in scores]
max_scores = np.array([100, 100, 50, 100, 50, 35, 35, 35, 35, 42])
scores = np.array(scores) / max_scores
weights = np.array([1/6, 1/15, 1/30, 1/30, 1/30, 1/9, 1/9, 1/9, 1/9, 2/9])