function [bound] = MonotoneChainBound(algs, windowSize, maxTime)
    [numAlgs numObjects] = size(algs);
    % maxTime + windowSize - 1 должно быть не больше numObjects
    bound = zeros(1, T - 1);
    algChoiceProbs = zeros(1, numAlgs);
    for t = 1 : maxTime
        
    end
    
end