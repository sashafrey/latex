function [algs, bestProb] = GreedyFindNextAlgorithm...
                            (algs, trainSize, errorLevel, minOrMaxOverfitProbabilty, maxTrainError)
    [numAlgs sampleSize] = size(algs);
    candidates = nchoosek(1:sampleSize, errorLevel);
    bestAlgorithm = zeros(1, sampleSize);
    
    bestProb = 1;
    if (~minOrMaxOverfitProbabilty)
        bestProb = 0;
    end
    useInclusionExclusion = false;
    for n = 1:size(candidates, 1)
        curAlgorithm = zeros(1, sampleSize);
        curAlgorithm( candidates(n, :) ) = 1;
        
        isNewAlgorithm = true;
        for s = 1:numAlgs
            if all(algs(s, :) == curAlgorithm)
                isNewAlgorithm = false;
            end
        end
        if (~isNewAlgorithm)
            continue;
        end
        if useInclusionExclusion
            prob = ComputeOverfitProbViaInclusionExclusion([algs; curAlgorithm], trainSize, errorLevel);
        else
            prob = ExactFunctional([algs; curAlgorithm], trainSize, ...
                                   MaxTrainErrorToEpsilon(sampleSize, trainSize, errorLevel, maxTrainError));
        end
        if ( (minOrMaxOverfitProbabilty && prob < bestProb) || (~minOrMaxOverfitProbabilty && prob > bestProb))
            bestProb = prob;
            bestAlgorithm = curAlgorithm;
        end
    end
    algs = [algs; bestAlgorithm];
    
end