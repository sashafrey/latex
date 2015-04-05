% maxTime = 200;
% numObjects = maxTime;
% goodErrorRate = 0.05;
% badErrorRate = 0.95;
% maxDist = 15;
% algs = zeros(2, numObjects);
% for n = 1 : numObjects
%     if 2 * n <= numObjects
%         algs(1, n) = rand() < goodErrorRate;
%         algs(2, n) = rand() < badErrorRate;
%     else
%         algs(1, n) = rand() < badErrorRate;
%         algs(2, n) = rand() < goodErrorRate;
%     end
% end

maxTime = 10;
maxDist = 10;
numObjects = 200;

[sample, sampleClasses] = GenerateSimpleSample(numObjects);
%[sample, sampleClasses] = GenerateCloseClasses(numObjects);
p = randperm(numObjects);
sample = sample(p, :);
sampleClasses = sampleClasses(p);
[graph, algs] = BuildLinearSet(sample, sampleClasses);
PaintSample(sample, sampleClasses);
%PaintAlgorithmsFamily(algs);
overfitProb = SCBound(algs, graph, GetBoundedDistanceDistribution(maxDist, numObjects))
%overfitProb = SCBound(algs, graph, GetGaussianDistanceDistribution(10, numObjects));
%overfitProb = TwoAlgorithmsBound(algs, GetGaussianDistanceDistribution(10, numObjects));
grid on
plot(overfitProb, 'b');
