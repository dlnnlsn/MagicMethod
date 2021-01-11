function normDiff(oldvals, newvals) {
  var total = 0
  for (var i = 0; i < oldvals.length; i++) {
    total += Math.pow(newvals[i] - oldvals[i], 2)
  }
  return Math.sqrt(total)
}

function isNumeric(n) {
  return !isNaN(parseFloat(n)) && isFinite(n);
}

function difficulties_from_scores(scores, actual_scores, test_weights) {
  var students = actual_scores.length
  var tests = actual_scores[0].length

  if (test_weights.length === 1 && Array.isArray(test_weights[0])) {
    test_weights = test_weights[0]
  }
  test_weights = test_weights.map(parseFloat)
  if (scores.length === 1 && Array.isArray(scores[0])) {
    scores = scores[0]
  }
  scores = scores.map(parseFloat)

  var numerators = Array(tests)
  var denominators = Array(tests)

  for (var j = 0; j < tests; ++j) {
    var numerator = 0
    var denominator = 0
    for (var i = 0; i < students; ++i) {
      if (isNumeric(actual_scores[i][j])) {
        numerator += test_weights[j] * scores[i] * parseFloat(actual_scores[i][j])
        denominator += test_weights[j] * Math.pow(scores[i], 2)
      }
    }
    numerators[j] = numerator
    denominators[j] = denominator
  }

  var lambda_coeff = 0
  var lambda = 1

  for (var j = 0; j < tests; ++j) {
    lambda_coeff += 1/denominators[j]
    lambda -= numerators[j] / denominators[j]
  }

  lambda /= lambda_coeff

  var difficulties = Array(tests)
  for (var j = 0; j < tests; ++j) {
    difficulties[j] = (numerators[j] + lambda) / denominators[j]
  }

  return difficulties
}

function scores_from_difficulties(difficulties, actual_scores, test_weights) {
  var students = actual_scores.length
  var tests = actual_scores[0].length

  if (test_weights.length === 1 && Array.isArray(test_weights[0])) {
    test_weights = test_weights[0]
  }
  test_weights = test_weights.map(parseFloat)
  if (difficulties.length === 1 && Array.isArray(difficulties[0])) {
    difficulties = difficulties[0]
  }
  difficulties = difficulties.map(parseFloat)

  var scores = Array(students)

  for (var i = 0; i < students; i++) {
    var numerator = 0
    var denominator = 0
    for (var j = 0; j < tests; j++) {
      if (isNumeric(actual_scores[i][j])) {
        numerator += test_weights[j] * difficulties[j] * parseFloat(actual_scores[i][j])
        denominator += test_weights[j] * Math.pow(difficulties[j], 2)
      }
    }
    scores[i] = numerator / denominator
  }

  return scores
}

function magicmethod(actual_scores, test_weights, tolerance=1e-3) {
  var tests = actual_scores[0].length

  if (test_weights.length === 1 && Array.isArray(test_weights[0])) {
    test_weights = test_weights[0]
  }
  test_weights = test_weights.map(parseFloat)

  var difficulties = Array(tests)
  for (var j = 0; j < tests; ++j) {
    difficulties[j] = 1/tests
  }

  var old_difficulties
  var scores

  do {
    scores = scores_from_difficulties(difficulties, actual_scores, test_weights)
    old_difficulties = difficulties
    difficulties = difficulties_from_scores(scores, actual_scores, test_weights)
  }
  while (normDiff(old_difficulties, difficulties) > tolerance)

  return [scores, difficulties]
}

function magicscores(actual_scores, test_weights, tolerance=1e-3) {
  var result = magicmethod(actual_scores, test_weights, tolerance)
  return result[0]
}

function magicdifficulties(actual_scores, test_weights, tolerance=1e-3) {
  var result = magicmethod(actual_scores, test_weights, tolerance)
  return result[1]
}
