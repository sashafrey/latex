function [answer] = ComputeFractionOf0_1SequencesWithFixedCountOfZerosInARow(sequenceLength, numOnes, numZerosInRow)
	
	
	chooseTable = zeros(sequenceLength + 1, numOnes + 1);
	chooseTable(:, 1) = 1;
	for i = 1:sequenceLength
		for j = 1:numOnes
			chooseTable(i + 1, j + 1) = chooseTable(i, j) + chooseTable(i, j + 1);
		end
	end
	
	
	dpArray = zeros(sequenceLength + 2, numOnes + 2);
	dpArray(1, 1) = 0;
	for subsequenceLength = 1:sequenceLength+1
		for onesUsed = 1:numOnes+1
			if (subsequenceLength > numZerosInRow)
				dpArray(subsequenceLength + 1, onesUsed + 1) = chooseTable(subsequenceLength - numZerosInRow ,onesUsed);
			end
			
			for lastOne = max(0,subsequenceLength - numZerosInRow) : subsequenceLength - 1
				dpArray(subsequenceLength + 1, onesUsed + 1) = dpArray(subsequenceLength + 1, onesUsed + 1) + ...
					dpArray(lastOne + 1, onesUsed);
			end
		end
	end
	
	answer = dpArray(sequenceLength + 2, numOnes + 2);
	
	%allSequences = nchoosek(1:sequenceLength, numOnes);
% 	testAnswer = sum(sum( ([allSequences (sequenceLength + 1) * ones(size ( allSequences, 1) , 1)] ...
% 		-  [ zeros(size ( allSequences, 1), 1) allSequences ]) > numZerosInRow, 2) > 0);
% 	
 	sequenceLength
 	numOnes
 	numZerosInRow
% 	
% 	if (answer ~= testAnswer)
% 		'DP algorithm fail!!!'
% 		%'My answer = '
% 		answer
% 		testAnswer
% 		
% 	end
	

	
	answer = answer / chooseTable(sequenceLength + 1, numOnes + 1);
	
	
end