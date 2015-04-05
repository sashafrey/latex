%mex  -output D:\Study\Science\Ёксперименты\AlgorithmSetsBuilders\BuildConjunctionSetNew D:\Study\Science\эксперименты\AlgorithmSetsBuilders\BuildLinearSet\build_conjunction_set.cpp
% %scored_reads = load('D:\Inform\TopCoder\MarathonMatches\MinorityVariants\MinorityVariants\scored_reads.txt');
% %sample = haberman(:, 1 : 3);
% %sampleClasses = haberman(:, 4)';
% %sampleSize = numel(sampleClasses)
sampleSize = 300;
ind = randsample(size(scored_reads, 1), sampleSize);
sample = [scored_reads(ind, 2 : 3)];
sampleClasses = scored_reads(ind, 1)';
trainSize = floor(sampleSize / 2);
 %[sample, sampleClasses] = GenerateSimpleSampleWithRandomNoise(sampleSize, 2, 10);
[graph, algs] = BuildConjunctionSetNew(sample, sampleClasses, 50, 20);
for n = 1 : numel(graph)
     graph{n} = graph{n}';
end
 
profile = ComputeInferiorityProfile(algs, graph);
pmqTable = ComputeExtendedPmqTable(sampleSize, trainSize, 0.13, size(profile, 3), min(sum(algs, 2)));
levels = zeros(1, sampleSize);
for n = 1 : size(profile, 1)
    levels(n) = sum(sum(profile(:, n, :) .* pmqTable(1 : size(profile, 1), n, 1 : size(profile, 3))));
end
hold on
plot([1 : sampleSize], levels, 'r');
plot([1 : sampleSize], cumsum(levels), 'b');

% for n = 1 : numel(eps)
%     n
%     res1(n) = ComputeGeneralizedGraphBound(algs, generalizedGraph, eps(n), trainSize);
%     res2(n) = ComputeGeneralizedGraphBound(newAlgs, newGeneralizedGraph, eps(n), trainSize);
%     %res2(n) = ComputeExtendedSCBound(ComputeExtendedPmqTable(sampleSize, trainSize, eps(n), sampleSize), profile);
% end

% hold on
% plot(eps, res1, 'b', 'LineWidth', 2);
% plot(eps, res2, 'r', 'LineWidth', 2);