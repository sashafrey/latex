function[overfitProb, expectedRisk] = MonteCarloEstimation(algs, trainSize, eps, numPartitions)
    if nargin < 4
        numPartitions = 20000;
    end
    [numAlgs sampleSize] = size(algs);
    totalError =  sum(algs, 2);
    thresholds = TrainErrorOverfitThreshold(sampleSize, trainSize, [0:sampleSize]', eps);
    overfitProb = zeros(size(eps));   
    expectedRisk = 0; 
    
    for j = 1:numPartitions
        ind = randperm(sampleSize);
        ind = ind(1:trainSize);
        observedError = sum(algs(:, ind), 2);
        minObservedError = min(observedError);
        maxTotalError = max(totalError(observedError == minObservedError));
        overfitProb = overfitProb + (thresholds(maxTotalError + 1, :) >= minObservedError);
        expectedRisk = expectedRisk + (maxTotalError - minObservedError) / (sampleSize - trainSize);
    end
    
    overfitProb = overfitProb / numPartitions;
    expectedRisk = expectedRisk / numPartitions;
end