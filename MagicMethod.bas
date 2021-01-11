REM  *****  BASIC  *****

function square(x)
	square = x * x
end function

function norm(oldvals, newvals)
	total = 0
	if ubound(oldvals) = 1 then
		for i = 1 to ubound(oldvals, 2)
			total = total + square(newvals(1, i) - oldvals(1, i))
		next i
	else
		for i = 1 to ubound(oldvals)
			total = total + square(newvals(i, 1) - oldvals(i, 1))
		next i
	end if
	norm = total
end function

function difficulties_from_scores(ideal_scores, actual_scores, test_weights)
	students = ubound(actual_scores)
	tests = ubound(actual_scores, 2)
	
	dim diff(1 to tests)
	dim numerators(1 to tests)
	dim denominators(1 to tests)
	
	for j = 1 to tests
		numerator = 0
		denominator = 0
		for i = 1 to students
			if isnumeric(actual_scores(i, j)) then
				numerator = numerator + test_weights(j) * ideal_scores(i) * actual_scores(i, j)
				denominator = denominator + test_weights(j) * square(ideal_scores(i))
			end if
		next i
		numerators(j) = numerator
		denominators(j) = denominator
	next j
	
	lambda_coeff = 0
	lambda = 1
	
	for j = 1 to tests
		lambda = lambda - numerators(j) / denominators(j)
		lambda_coeff = lambda_coeff + 1/denominators(j)
	next j
	
	lambda = lambda / lambda_coeff
	
	for j = 1 to tests
		diff(j) = (numerators(j) + lambda) / denominators(j)
	next j
	
	difficulties_from_scores = diff
end function

function scores_from_difficulties(difficulties, actual_scores, test_weights)
	students = ubound(actual_scores)
	tests = ubound(actual_scores, 2)
	
	dim scores(1 to students)
	
	for i = 1 to students
		numerator = 0
		denominator = 0
		for j = 1 to tests
			if isnumeric(actual_scores(i, j)) then
				numerator = numerator + test_weights(j) * difficulties(j) * actual_scores(i, j)
				denominator = denominator + test_weights(j) * square(difficulties(j))
			end if
		next j
		scores(i) = numerator / denominator
	next i
	
	scores_from_difficulties = scores
end function

function magicmethod(actual_scores, test_weights, optional tolerance)
	if ismissing(tolerance) then
		tolerance = 1e-3
	end if
	
	tolerance = square(tolerance)

	students = ubound(actual_scores)
	tests = ubound(actual_scores, 2)
	
	dim weights(1 to tests)
	
	if ubound(test_weights) = 1 then
		for j = 1 to tests
			weights(j) = test_weights(1, j)
		next j
	else
		for j = 1 to tests
			weights(j) = test_weights(j, 1)
		next j
	end if
	
	dim result(1 to 2)
	dim difficulties(1 to tests)

	for j = 1 to tests
		difficulties(j) = 1/tests
	next j
	
	do
		scores = scores_from_difficulties(difficulties, actual_scores, weights)
		old_difficulties = difficulties
		difficulties = difficulties_from_scores(scores, actual_scores, weights)
	loop while norm(old_difficulties, difficulties) > tolerance
	
	result(1) = scores
	result(2) = difficulties
	
	magicmethod = result	
end function

function magicscores(actual_scores, test_weights, optional tolerance)
	magicmethodresult = magicmethod(actual_scores, test_weights, tolerance)
	magicscores = magicmethodresult(1)
end function

function magicdifficulties(actual_scores, test_weights, optional tolerance)
	magicmethodresult = magicmethod(actual_scores, test_weights, tolerance)
	magicdifficulties = magicmethodresult(2)
end function
