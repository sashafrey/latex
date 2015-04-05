function [algs] = Build1NNSet(sample, sampleClasses, l);

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

algs = zeros(N, L);
for i = 1:N
	[t, minPos] = min(dist(ind(i, :), :) ) ;
	algs(i, :) = (sampleClasses ~= sampleClasses( ind(i, minPos) ) ); 
end

algs = unique(algs, 'rows');

end