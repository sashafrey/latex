algs = [1 1 1 1 0 0 0 0 0 0 0 0];
trainSize = 6;
errorLevel = 4;
prob = [0];

while (prob(numel(prob)) < 1)
    [algs, prob(numel(prob) + 1)] = GreedyFindNextAlgorithm(algs, trainSize, errorLevel, true, 1);
end
algs 
prob
