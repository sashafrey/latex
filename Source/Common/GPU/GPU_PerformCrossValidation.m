function [QEps, eps, trainErrorRate, testErrorRate] = GPU_PerformCrossValidation(ev, ec, nIters, seed, nTrainItems)
    % GPU_PerformCrossValidation
    % Runs numeric Monte-Carlo to estimate the bias between test and train
    % error rate of pessimistic error risk minimizer.
    %
    % Usage: 
    % [QEps, eps, overfitting] = GPU_PerformCrossValidation(ev, ec, nIters, randomSeed),
    % where
    %   - ev is a matrix of size [nItems * nAlgs] that describes error
    %   vectors,
    %   - ec is a vector of length [nAlgs] that contains error counts,
    %   - nIters is the number of monte-carlo iterations to perform,
    %   - seed is an number defining behaviour of random number generator.
    %   Optional, default = 0 (time-initialized random number).
    %   - nTrainItems - number of items for training ($\ell$)
    %   - QEps is a vector of the same size as eps. QEps(i) gives the
    %   tail probability of the deviation on test and train sample larger
    %   than eps(i).
    %   - trainErrorRate is a vector of length nIters that gives train
    %   error rate of pessimistic ERM for each iteration
    %   - testErrorRate is a vector of length nIters that gives test
    %   error rate of pessimistic ERM for each iteration

    if (~exist('seed', 'var'))
        seed = 0;
    end
    
    nItems = size(ev, 1);
    nAlgs = size(ev, 2);
    
    if (~exist('nTrainItems', 'var'))
        nTrainItems = floor(nItems / 2);
    end

    Check(nAlgs > 0);
    Check(nItems > 0);
    Check(nTrainItems > 0);
    Check(nTrainItems < nItems);
    Check(length(ec) == nAlgs);
    trainEC = zeros(nIters, 1, 'int32');
    testEC = zeros(nIters, 1, 'int32');
    [errco, ev1, ec1, trainEC, testEC] = calllib(GPU_LibName, 'performCrossValidation', ev, ec, nItems, int32(nTrainItems), nAlgs, nIters, seed, trainEC, testEC);
    GPU_CheckErrCode(errco);
    
    nEpsValues = floor(nItems / 4);
    trainErrorRate = double(trainEC)/(nTrainItems);
    testErrorRate = double(testEC)/(nItems - nTrainItems);
    overfitting = testErrorRate - trainErrorRate;
    eps = zeros(1, nEpsValues);
    QEps = zeros(1, nEpsValues);
    for i=1:nEpsValues
        eps(i) = double(i-1) * 4 / double(nItems);
        QEps(i) = sum(overfitting >= eps(i)) / length(overfitting);
    end
end

