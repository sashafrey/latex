eps = 0.05
maxn = 500
res = zeros(1, maxn);
cTable = ComputeChooseTable(1000);
monteCarlo = zeros(1, maxn)
for trainSize = 1:maxn
    sampleSize = 2 * trainSize
    res(trainSize) = ERMMinimaxLowerBound(sampleSize, trainSize, eps, 10, cTable);
    [sample, sampleClasses] = GenerateSimpleSample(sampleSize);
    %algs = BuildLinearSet(sample, sampleClasses);
    %monteCarlo(trainSize) = MonteCarloEstimation(algs, trainSize, eps, 20000);
end
     
hold on
grid on
plot(2 * [1 : maxn], res, 'b')
plot(2 * [1 : maxn], monteCarlo, 'g')
grid on
    