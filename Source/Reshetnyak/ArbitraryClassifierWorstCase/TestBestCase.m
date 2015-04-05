sampleSize = 200;
trainSize = sampleSize / 2;
%numAlgs = 2;
errorLevel = 5;

% algs = BuildLevel(sampleSize, errorLevel);
% numAlgs = size(algs, 1);
% prob1 = zeros(1,numAlgs);
% for n = 1 :numAlgs
%     prob1(n) = ComputeOverfitProbabiltyForLexicographicallyFirstSet(n, sampleSize, trainSize, errorLevel);
% end
numSteps = 40;
step = 10;
offset  = 100;
prob = zeros(1, numSteps);
eps = 0.05;
chooseTable = ComputeChooseTable(offset + step * numSteps);
overfitProbUnionBound = zeros(1, numSteps);
for n = 1:numSteps;
    sampleSize = offset + step * n;
    trainSize = sampleSize / 2;
    prob(n) = ComputeOverfitProbabiltyForLexicographicallyFirstSet(sampleSize^2, sampleSize, trainSize, ...
        ceil(eps * trainSize), chooseTable );
    overfitProbUnionBound(n) = sampleSize^2 * chooseTable(sampleSize + 1 - ceil(eps * trainSize), trainSize) / ...
        chooseTable(sampleSize + 1, trainSize);
end

sampleSizes = offset + step * [1 : numSteps];
hold on
plot(sampleSizes , prob);
plot( sampleSizes, overfitProbUnionBound, 'r');
% prob2 = ExactFunctional(algs, trainSize, (errorLevel - 0.5) / (sampleSize - trainSize) )
% 
% if sum(prob1 ~= prob2) > 0
%     'ERROR!!!!'
%     find(prob1 ~= prob2)
% else
%     'OK!!!'
% end
