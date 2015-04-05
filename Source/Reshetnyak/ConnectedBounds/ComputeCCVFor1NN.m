function [ccv] = ComputeCCVFor1NN(sample, sampleClasses, etalons, testSize)
	numEtalons = numel(etalons);
	numObjects = size(sample, 1);
	
	dist = zeros(numObjects, numEtalons);
	for i = 1:numObjects
		for j = 1:numEtalons
			dist(i, j) = sum( (sample(i, :) - sample(etalons(j), :) ).^2 );
		end
	end
	
	[temp, ind] = sort(dist, 2);
	
	ccv = 0;
	for m = 1:min(testSize, numEtalons)
		factor = 1/testSize * nchoosek(numObjects - m - 1, testSize - m);
		ccv = ccv + factor * sum(sampleClasses ~=  sampleClasses( etalons( ind(:, m) ) ) );
	end
	
	ccv = ccv/ nchoosek(numObjects, testSize);
	
end