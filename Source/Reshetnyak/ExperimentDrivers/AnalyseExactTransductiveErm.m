sampleSize = 16;
%trainSize = 25
trainSize = sampleSize / 2;
eps = 0.1;
[sample sampleClasses] = GenerateSimpleSample(sampleSize);

algs = BuildLinearSet(sample, zeros(1, sampleSize));
t = randperm(sampleSize);
trainSample = t(1:trainSize);
[bound, optClassification] = TransductiveExactERM(algs, trainSample, sampleClasses(trainSample), eps)

for n = 1:2
    figure
    hold on
    scatter(sample(:,1), sample(:, 2), 30, optClassification(n, :)', 'filled');
    scatter(sample(trainSample,1), sample(trainSample, 2), 40, optClassification(n, trainSample)', 's');
    hold off
    PaintAlgorithmsFamily(algs ~= repmat(optClassification(n, :),size(algs, 1), 1));
end