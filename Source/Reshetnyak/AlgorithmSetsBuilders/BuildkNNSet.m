function [algs] = BuildkNNSet(sample, sampleClasses, l, k);

L = size(sample, 1);
dist = zeros(L, L);
for i = 1:L
	for j = 1:L
		dist(i, j) = sum( (sample(i,:) - sample(j, :)).^2);
	end
	dist(i, i) = inf;
end

N = nchoosek(L, l);
ind = nchoosek([1:L], l);
numClasses = numel(unique(sampleClasses) );
sampleClasses(sampleClasses == -1) = 2;

algs = zeros(N, L);
[dist, ord] = sort(dist, 2);

usedObjects = zeros(L,k);
for i = 1:N
	
	curInd = false(1, L);
	curInd(ind(i, :) ) = true;
	
	
	for j = 1:L
		temp = ord(j, curInd(ord(j, :) ) );
		usedObjects(j, :) = temp(1:k);
	end
	
	estimate = zeros(numClasses, L);
	
	for n = 1:numClasses
		estimate(n, :) = sum(sampleClasses(usedObjects) == n, 2);
	end
	
	[temp, ans] = max(estimate);
	
	algs(i, :) = (sampleClasses ~= ans); 
end

algs = unique(algs, 'rows');

end