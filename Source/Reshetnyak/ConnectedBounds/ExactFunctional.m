function [overfitProb] = ExactFunctional(algs, trainSize, eps)
    [numAlgs sampleSize] = size(algs);
    ind = nchoosek(1:sampleSize, trainSize); 
    totalError =  sum(algs , 2);
    thresholds = TrainErrorOverfitThreshold(sampleSize, trainSize, [0:sampleSize], eps);
    overfitProb = 0;
    
    for j = 1:size(ind, 1)
        observedError = sum(algs(:, ind(j, :) ), 2);
        minObservedError = min(observedError);
        maxTotalError = max(totalError(observedError == minObservedError));
        if minObservedError <= thresholds(maxTotalError + 1)
            overfitProb = overfitProb + 1;
        end
    end
    
    overfitProb = overfitProb / nchoosek(sampleSize, trainSize);
    
end