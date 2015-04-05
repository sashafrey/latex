function  StudyKNN(L, l, eps, numEtalons, k)

if nargin < 5
	k = 1;
end
if nargin < 4
	numEtalons = 5;
end
if nargin < 3
	eps = 0.05;
end
if nargin < 1
	L = 40;
end
if nargin < 2
	l = L / 2;
end

%[sample, sampleClasses] = GenerateSimpleSample(L);
[sample, sampleClasses] = GenerateCloseClasses(L);

scatter(sample(:,1), sample(:, 2), 30, sampleClasses, 'filled');
algs = BuildkNNwithEtalonsSet(sample, sampleClasses, l, k, numEtalons);
PaintAlgorithmsFamily(algs);



end