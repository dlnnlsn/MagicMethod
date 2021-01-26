from piecewise_linear_difficulties import *

def student_scores(students, optimize_output):
    return optimize_output.x[:students]

def ms_and_bs(students, points_per_difficulty_function, optimize_output):
    for index in range(students, len(optimize_output.x), 2 * points_per_difficulty_function):
        yield optimize_output.x[index : index + points_per_difficulty_function], optimize_output.x[index + points_per_difficulty_function : index + 2 * points_per_difficulty_function]

def ms(students, points_per_difficulty_function, optimize_output):
    for m, _ in ms_and_bs(students, points_per_difficulty_function, optimize_output):
        yield m

def bs(students, points_per_difficulty_function, optimize_output):
    for _, b in ms_and_bs(students, points_per_difficulty_function, optimize_output):
        yield b

def predicted_scores(students, points_per_difficulty_function, optimize_output):
    ideal_scores = student_scores(students, optimize_output)
    difficulty_functions = starmap(difficulty_function, ms_and_bs(students, points_per_difficulty_function, optimize_output))
    return np.array([list(map(diff, ideal_scores)) for diff in difficulty_functions]).transpose()

def draw_difficulty_function(plt, ms, bs, *args, **kwargs):
    xs = [0] + list(bs)
    ys = [0] + [m * b for m, b in zip(ms, bs)]
    for i in range(1, len(ys)):
        xs[i] += xs[i - 1]
        ys[i] += ys[i - 1]
    plt.plot(xs, ys, *args, **kwargs)