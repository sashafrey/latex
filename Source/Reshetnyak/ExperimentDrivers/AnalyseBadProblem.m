sampleSize = 400;
trainSize = sampleSize / 2;
eps = 0.1;
[sample sampleClasses] = GenerateSimpleSample(sampleSize);

algs = BuildLinearSet(sample, zeros(1, sampleSize));
t = randperm(sampleSize);
trainSample = t(1:trainSize);
sampleClasses(trainSample) = -sampleClasses(trainSample);
%bound = MonteCarloEstimation(algs, trainSize, eps, 50000)

hold on
scatter(sample(:,1), sample(:, 2), 30, sampleClasses, 'filled');
scatter(sample(trainSample, 1), sample(trainSample, 2), 40, sampleClasses(trainSample), 's');
