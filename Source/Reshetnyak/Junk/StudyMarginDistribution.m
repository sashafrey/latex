sampleSize = 400;
eps = 0.05;
trainSize = sampleSize / 2;
%[sample, sampleClasses] = GenerateSimpleSample(sampleSize, 2);
%[sample, sampleClasses] = GenerateCloseClasses(sampleSize, 3);
%[sample, sampleClasses] = GenerateSimpleSampleWithRandomNoise(sampleSize, 2, 20);
%[familyGraph, algs] = BuildLinearSet(sample, sampleClasses);
%margins = ComputeMarginDistribution(algs(1, :), algs);
%scProfile = ComputeScProfile(1, familyGraph);
%splittingProfile = sum(scProfile, 2);
m = ceil(eps * (sampleSize - trainSize))
borderObjects = sum(margins <= m);

hold on
plot([0 : borderObjects], hygepdf([0 : borderObjects], sampleSize, borderObjects, trainSize), 'r');
plot([0 : borderObjects], splittingProfile(m + 1) * nchoosek(sampleSize - m, trainSize) / nchoosek(sampleSize, trainSize)...
     * hygepdf([0 : borderObjects], sampleSize - m, borderObjects, trainSize));
grid on



