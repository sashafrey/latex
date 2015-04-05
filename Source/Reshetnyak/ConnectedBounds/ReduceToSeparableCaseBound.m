sampleSize = 200;
trainSize = sampleSize / 2;

[sample, sampleClasses] = GenerateCloseClasses(sampleSize);
%[sample, sampleClasses] = GenerateSimpleSampleWithRandomNoise(sampleSize, 2, 15);
%[sample, sampleClasses] = GenerateSimpleSample(sampleSize);
[graph, algs] = BuildLinearSet(sample, sampleClasses);
shift = sum(algs(1, :))
if (shift == 0)
    return 
end

PaintSample(sample, sampleClasses);
PaintAlgorithmsFamily(algs, 10)

scProfile = ComputeExtendedProfile(graph, sum(algs, 2));
inferiorityProfile = ComputeInferiorityProfile(algs, graph);



for n = size(algs, 1) : -1 : 1
    algs(n, :) = xor( algs(n, :), algs(1, :));
end
terr = sum(algs, 2);
[terr, ind] = sort(terr);
algs = algs(ind, :);
reducedProfile = ComputeScProfile(1, BuildFamilyGraph(algs ));

eps = [0.01:0.01:0.15];
scBound = zeros(size(eps));
newBound = zeros(size(eps));
infBound = zeros(size(eps));


for n = 1 : numel(eps)
    prmqTable = ComputeExtendedPmqTable(sampleSize, trainSize, eps(n), size(scProfile, 3) - 1);
    n
    scBound(n) = ComputeExtendedSCBound(prmqTable, scProfile);
    infBound(n) = ComputeExtendedSCBound(prmqTable, inferiorityProfile);
	[worstProb, mNew, qNew] = ComputeWorstCaseRivalTable(shift, sampleSize, trainSize, eps(n), size(reducedProfile, 2) -1);
    newBound(n) = hygecdf(TrainErrorOverfitThreshold(sampleSize, trainSize, shift, eps(n)), sampleSize, shift, trainSize) + ... 
                  sum( sum(reducedProfile .* worstProb));
end

figure
hold on
grid on
set(gca, 'xtick', 0.01:0.01:0.25)
plot(eps, scBound, 'b', 'LineWidth', 2)
plot(eps, newBound, 'r', 'LineWidth', 2)
plot(eps, infBound, 'g', 'LineWidth', 2)
%plot(eps, min(monteCarloBound, 'k', 'LineWidth', 2)
scBound
newBound
infBound