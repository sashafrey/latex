function [ccv, overfitProb, algs] = ComputeCCVForkNN(k, sample, sampleClasses, etalons, testSize)
	
	L = size(sample, 1);
	numEtalons = numel(etalons);
	numClasses = numel(unique(sampleClasses) );
	
	dist = zeros(L, L);
	for i = 1:L
		for j = 1:L
			dist(i, j) = sum( (sample(i, :) - sample(j, :) ).^2 );
		end
	end
	
	
	algs = zeros(2^numEtalons - 1, L);
	algNumber = 0;
	overfitProb = 0;
	
	threshold =  floor( (L - testSize)/L * ([1:L] - eps * testSize) );
	ccv = 0;
	for count = 1:numEtalons
		
		etalonSets = nchoosek(etalons, count);
		curWeight = nchoosek(L - numEtalons - 1, L - testSize - count);
		coeff = nchoosek(L - numEtalons, L - testSize - count) * ...
					hygecdf(threshold, L - numEtalons, [1:L], L  - testSize - count);
		
		for n = 1:size(etalonSets, 1)	
			[temp, ord] = sort( dist(:, etalonSets(n, :) ), 2);
			ord = ord(:, 1:k);
			
			estimate = zeros(numClasses, L);
			for c = 1:numClasses
				if (k == 1)
					estimate(c, :) = (sampleClasses(etalonSets(n, ord) ) == c);
				else
					estimate(c, :) = sum(sampleClasses(etalonSets(n, ord)) == c, 1);
				end
			end
		
			[temp, ans] = max(estimate);
			algNumber = algNumber + 1;
			algs(algNumber, :) = (sampleClasses ~= ans); 
			m = sum(algs(algNumber, :) ) + sum( algs(algNumber, etalonSets(n, :) ) ) - ...
				sum( algs(algNumber, etalons) );
			if (m > 0)
				overfitProb = overfitProb + coeff(m);
			end
			ccv = ccv + m / testSize * curWeight;
		end
	end
	
	algs = unique(algs, 'rows');
	ccv = ccv / nchoosek(L, testSize);
	overfitProb = overfitProb / nchoosek(L, testSize);
	
end