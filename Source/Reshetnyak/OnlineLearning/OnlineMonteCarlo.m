function [qualityFunctional] = OnlineMonteCarlo(algs, CalcFunctional, numPermutations)
    if nargin < 3
        numPermutations = 20000;
    end
    
    [numAlgs sampleSize] = size(algs);
    totalError =  sum(algs, 2);
    qualityFunctional = zeros(1, sampleSize - 1);   
    
    for n = 1:numPermutations
        objectsOrder = randperm(sampleSize);
        cumError = cumsum(algs(:,objectsOrder), 2);
        for t = 1 : sampleSize - 1
            qualityFunctional(t) = qualityFunctional(t) + ... 
                CalcFunctional(totalError, cumError(:, t), algs(:, objectsOrder(t + 1)));
        end
    end

    qualityFunctional = qualityFunctional / numPermutations;
end
