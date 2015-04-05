function StudykNNEtalonSet(L, l, eps, numEtalons, k) 
	
function [etalons] = BuildEtalonSet(numEtalons)
	etalons = 1:numEtalons;
	while ( (numel( unique( sampleClasses(etalons) ) ) ~= numClasses)  ) %|| ...
			%(sum(sampleClasses(etalons) == 1) == 1) || (sum(sampleClasses(etalons) == 1) ==  numEtalons - 1) )
		etalons = randperm(L - 1);
		etalons = etalons(1: numEtalons);
	end
end

if nargin < 5
	k = 1;
end
if nargin < 4
	numEtalons =2;
end
if nargin < 3
	eps = 0.05;
end
if nargin < 1
	L = 10;
end
if nargin < 2
	l = floor(L / 2);
end

[sample, sampleClasses] = GenerateSimpleSample(L);
%sample(1, :) = [100 100];
%sample


numIterations = 1
%[sample, sampleClasses] = GenerateCloseClasses(L);


sampleClasses(sampleClasses == -1) = 2;
sampleClasses(L) = 1;
numClasses = numel(unique(sampleClasses) );


worstEtalons = zeros(1, numEtalons);
worstCCV = 0;
worstOverfitProb = 1;
for numTries = 1:numIterations
	etalons = BuildEtalonSet(numEtalons);
	[ccv, overfitProb] = ComputeCCVForkNN(1, sample, sampleClasses, etalons, L - l);
	
	if (ccv > worstCCV)
		worstCCV = ccv;
		worstEtalons = etalons;
		worstOverfitProb = overfitProb;
	end
end
	
%numEtalons = numEtalons + 1;


bestEtalons = zeros(1, numEtalons);
bestCCV = 1;

for numTries = 1:numIterations
	etalons = [BuildEtalonSet(numEtalons) L];
	[ccv, overfitProb] = ComputeCCVForkNN(1, sample, sampleClasses, etalons, L - l);
	
	if (ccv < bestCCV)
		bestCCV = ccv;
		bestEtalons = etalons;
	end
end


'Worst CCV ='
worstCCV
'Best CCV = '
bestCCV

'Overfit probabilty = '
worstOverfitProb

if (worstCCV > bestCCV + 0.01)
	'Bad case found!!!'
end

hold on
scatter(sample(:,1), sample(:, 2), 20, sampleClasses', 'filled');
scatter(sample(worstEtalons, 1), sample(worstEtalons, 2), 50, sampleClasses(worstEtalons)', 's');
hold off

figure
hold on
scatter(sample(:,1), sample(:, 2), 20, sampleClasses', 'filled');
scatter(sample(bestEtalons, 1), sample(bestEtalons, 2), 50, sampleClasses(bestEtalons)', 's');
hold off


% ccv1 = ComputeCCVFor1NN(sample, sampleClasses, etalons, l);
% 'CompleteCrossValidation, first approach:'
% ccv1

%ccv2 = ComputeCCVForKNN(k, sample, sampleClasses, etalons, L - l);
%'CompleteCrossValidation, second approach:'
%ccv2

%PaintAlgorithmsFamily(algs);
%size(algs, 1)


end