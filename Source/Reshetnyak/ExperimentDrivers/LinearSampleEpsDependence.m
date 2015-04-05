sampleSize = 200;
eps = [0.01:0.01:0.25];

vapnikBound = zeros(size(eps));
vapnikStratBound = zeros(size(eps));
ermScBound = zeros(size(eps));
transBound = zeros(numel(eps), 2);
connectivityBound = zeros(size(eps));
uniformBonferonniBound = zeros(size(eps));
%ermStratificationBound = zeros(size(eps));
%monteCarlo = zeros(size(eps));
trainSize = sampleSize / 2
testSize = sampleSize - trainSize;

[sample, sampleClasses] = GenerateSimpleSample(sampleSize);
%[sample, sampleClasses] = GenerateCloseClasses(sampleSize);
figure
scatter(sample(:,1), sample(:, 2), 50, sampleClasses, 'filled');
algs = BuildLinearSet(sample, sampleClasses);
graph = BuildFamilyGraph(algs);
%vicinitySizes = cellfun('length', graph);
%PaintAlgorithmsFamily(algs, 10);

 scProfile = ComputeScProfile(1, graph);
% extScProfile = ComputeExtendedProfile(graph, sum(algs, 2));
 stratProfile = sum(scProfile, 2);
% t = randperm(sampleSize);
% trainSample = t(1:trainSize);
  for n = 1:numel(eps)
%     extPmqTable = ComputeExtendedPmqTable(sampleSize, trainSize, eps(n), max(vicinitySizes));
%     %pmqTable = ComputePmqTable(sampleSize, trainSize, eps(n), 0, max(vicinitySizes));
%     connectivityBound(n) = ErmConnectivityBound(sampleSize, extPmqTable, vicinitySizes);
%     ermScBound(n) = ComputeExtendedSCBound(extPmqTable, extScProfile);
      uniformBonferonniBound(n) = ConnectedFunctional(algs, graph, trainSize, eps(n));
%      
%     %[transBound(n, :), optSource] = TransductivePessimisticERM(algs, trainSample, sampleClasses(trainSample), eps(n))
%     %ermStratificationBound(n) = ComputeStratificationBound(pmqTable, stratProfile);
     possibleErrorLevels = ceil(eps(n) * testSize) : sampleSize;
     hypergeometricCoeffs = hygecdf(TrainErrorOverfitThreshold(sampleSize, trainSize, possibleErrorLevels, eps(n)),...
                                    sampleSize, possibleErrorLevels, trainSize);
     vapnikBound(n) = size(algs, 1) * max(hypergeometricCoeffs);
     vapnikStratBound(n) = sum(hypergeometricCoeffs .* stratProfile(possibleErrorLevels)');
 end
%[transBound, optSource] = TransductivePessimisticERM(algs, trainSample, sampleClasses(trainSample), eps)
[monteCarlo expectedRisk] = MonteCarloEstimation(algs, trainSize, eps, 20000);
figure
% hold on
%  plot(eps, vapnikBound, 'k');
% % plot(eps, connectivityBound, 'b');
%  plot(eps, vapnikStratBound, 'm');
%  plot(eps, uniformBonferonniBound, 'b');
% grid on
% figure
% hold on plot(eps, monteCarlo, 'b');
% plot(eps, ermScBound, 'r');
%plot(eps, ermStratificationBound, 'g');
% figure
% hold on
% grid on
% plot(eps, min(transBound(1, :), 1), 'r');
% plot(eps, min(transBound(2, :), 1), 'b');
% figure
% hold on
% scatter(sample(trainSample,1), sample(trainSample, 2), 40, algs(optSource(2,5), trainSample)', 's');
% scatter(sample(:,1), sample(:, 2), 30, algs(optSource(2, 5), :)', 'filled');
% grid on
% figure
% plot(eps, vapnikBound ./ connectivityBound, 'g');
% %if any(ermScBound <= monteCarlo) 
% %    'ERROR'
% %end
expectedRisk
