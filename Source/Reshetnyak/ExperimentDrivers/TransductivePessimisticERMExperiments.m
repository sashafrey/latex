points = [400];
bound = zeros(numel(points),2);
for n = 1:numel(points)
    sampleSize = points(n);
    trainSize = sampleSize / 2;
    eps = 0.05;
    [sample, sampleClasses] = GenerateSimpleSample(sampleSize);
    algs = BuildLinearSet(sample, zeros(1, sampleSize));
    t = randperm(sampleSize);
    trainSample = t(1:trainSize);
    [bound(n, :), optSource] = TransductivePessimisticERM(algs, trainSample, sampleClasses(trainSample), eps)
end

% hold on 
% plot(points, bound(:,1)', 'r');
% plot(points, bound(:,2)', 'b');
% hold off


for n = 1:2
    figure
    hold on
    scatter(sample(:,1), sample(:, 2), 30, algs(optSource(n), :)', 'filled');
    scatter(sample(trainSample,1), sample(trainSample, 2), 40, algs(optSource(n), trainSample)', 's');
    hold off
    PaintAlgorithmsFamily(algs ~= repmat(algs(optSource(n),:),size(algs, 1), 1), 20 );
end
