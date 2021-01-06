class MagicMethod():

    def __init__(self, tolerance=1e-3, max_iterations=100):
        self.tests = {}
        self.scores = {}
        self.student_scores = {}
        self.test_difficulties = {}
        self.dirty = False
        self.has_all_scores_and_difficulties = False
        self._badness = 0
        self.tolerance = tolerance
        self.max_iterations = max_iterations


    def add_student(self, student_name, test_scores = {}):
        for test in test_scores:
            if not test in self.tests:
                self.add_test(test)
        if not student_name in self.scores:
            self.scores[student_name] = {}
            self.has_all_scores_and_difficulties = False
        for test, score in test_scores.items():
            self.set_score(student_name, test, score)


    def add_test(self, test_name, test_weight=1):
        self.dirty = True
        if test_name in self.tests:
            if not test_weight is None:
                if self.tests[test_name] == test_weight:
                    self.dirty = False
                self.tests[test_name] = test_weight
            else:
                self.dirty = False
            return
        self.has_all_scores_and_difficulties = False
        self.tests[test_name] = test_weight or 1


    def set_score(self, student_name, test_name, test_score):
        if self.scores.get(student_name, {}).get(test_name) == test_score: return
        if not student_name in self.scores:
            self.add_student(student_name)
        self.dirty = True
        self.scores[student_name][test_name] = test_score


    def set_scores_for_test(self, test_name, scores, test_weight=None):
        self.add_test(test_name, test_weight)
        for student, score in scores.items():
            self.set_score(student, test_name, score)
    

    set_scores_for_student = add_student
    set_test_weight = add_test


    def set_initial_scores_and_difficulties(self):
        if self.has_all_scores_and_difficulties: return
        for student in self.scores:
            self.student_scores[student] = 1
        for test in self.tests:
            self.test_difficulties[test] = 1/len(self.tests)
        self.has_all_scores_and_difficulties = True


    def badness(self):
        if not self.dirty:
            return self._badness
        
        self.set_initial_scores_and_difficulties()
        self._badness = 0
        for test, weight in self.tests.items():
            for student, score in self.student_scores.items():
                actual_score = self.scores[student].get(test)
                if actual_score is not None:
                    self._badness += weight * (score * self.test_difficulties[test] - actual_score)**2
        return self._badness
            

    def update_difficulties_from_student_scores(self):
        test_numerators = {}
        test_denominators = {}
        for test, weight in self.tests.items():
            numerator = 0
            denominator = 0
            for student, actual_scores in self.scores.items():
                actual_score = actual_scores.get(test)
                if actual_score is not None:
                    ideal_score = self.student_scores[student]
                    numerator += weight * ideal_score * actual_score
                    denominator += weight * ideal_score**2
            test_numerators[test] = numerator
            test_denominators[test] = denominator
        lambda_coeff = sum(map(lambda x: 1/x, test_denominators.values()))
        lambda_rhs = 1
        for test, numerator in test_numerators:
            lambda_rhs -= numerator / test_denominators[test]
        lambda_value = lambda_rhs / lambda_coeff
        for test, numerator in test_numerators:
            self.test_difficulties[test] = (lambda_value + numerator) / test_denominators[test]


    def update_student_scores_from_difficulties(self):
        for student, actual_student_scores in self.scores.items():
            numerator = 0
            denominator = 0
            for test, weight in self.tests.items():
                score = actual_student_scores.get(test)
                if score is not None:
                    difficulty = self.test_difficulties[test]
                    numerator += weight * difficulty * score
                    denominator += weight * difficulty
            self.student_scores[student] = numerator / denominator if denominator else 1


    def calculate_scores_and_difficulties(self):
        if not self.dirty: return
        previous_badness = self.badness()
        for _ in range(self.max_iterations):
            self.update_student_scores_from_difficulties()
            self.update_difficulties_from_student_scores()
            if abs(self.badness() - previous_badness) < self.tolerance:
                break
            previous_badness = self.badness()
        self.dirty = False


    def get_difficulties(self):
        self.calculate_scores_and_difficulties()
        return self.test_difficulties


    def get_student_scores(self):
        self.calculate_scores_and_difficulties()
        return self.student_scores
        
