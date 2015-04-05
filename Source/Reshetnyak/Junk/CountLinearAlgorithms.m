function CountLinearAlgorithms()
	 L = 20;
	 numIter = 100;
	 count = zeros(1, numIter)
	 for n  = 1:numIter
		%L = 
		[sample, sampleClasses] = GenerateDiffClasses(L);
		%scatter(sample(:, 1), sample(:, 2), 25, sampleClasses, 'filled');
		algs = BuildLinearSet(sample, sampleClasses);
		
		count(n) = size(algs, 1);
		if (count(n) ~= L * (L - 1) + 2)
			scatter(sample(:, 1), sample(:, 2), 25, sampleClasses, 'filled');
		end
	 end
	 '„исло всевозможных раздел€ющих пр€мых'
	 count
	 if (numel(unique(count)) > 1)
		 '√ипотеза неверна'
	 end
	 
	 possValues = unique(count)
end