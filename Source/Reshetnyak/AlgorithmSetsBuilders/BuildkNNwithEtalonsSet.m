%
%

function [algs] = BuildkNNwithEtalonsSet(sample, sampleClasses, k, numEtalons);
%sample - L * dim матрица "объекты - признаки"
%sampleClasses - вектор классификации
%l - длина обучающей выборки
%k - число соседей, по которым производится классификация
%numEtalons - число используемых эталонов

L = size(sample, 1);
dist = zeros(L, L);
for i = 1:L
	for j = 1:L
		dist(i, j) = sum( (sample(i,:) - sample(j, :)).^2);
	end
	dist(i, i) = 0;
end

etalonSets = nchoosek([1:L], numEtalons);
numSets = nchoosek(L, numEtalons);

numClasses = numel(unique(sampleClasses) );
sampleClasses(sampleClasses == -1) = 2;

algs = zeros(numSets, L);

%usedObjects = zeros(L,k);
for n = 1:numSets
	[temp, ord] = sort(dist(:, etalonSets(n, :) ), 2);
	ord = ord(:, 1:k);
	
	estimate = zeros(numClasses, L);
	
	for c = 1:numClasses
		if (k == 1)
			estimate(c, :) = (sampleClasses(etalonSets(n, ord)) == c);
		else
			estimate(c, :) = sum(sampleClasses(etalonSets(n, ord)) == c, 1);
		end
	end
	
	[temp, ans] = max(estimate);
	
	algs(n, :) = (sampleClasses ~= ans); 
end

algs = unique(algs, 'rows');

end