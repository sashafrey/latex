function [bound] = ConnectedFunctional(algs, graph, trainSize, eps)
    [numAlgs sampleSize] = size(algs);
    errorLevel = sum(algs, 2);
    stratProfile = zeros(1, sampleSize);
    sources = [];
    for n = 1 : numAlgs
        isSource = true;
        for v = 1:numel(graph{n})
            if graph{n}(v) < n
                isSource = false;
            end
        end
        if isSource
            sources = [sources; n];
        elseif errorLevel(n) > 0
            stratProfile(errorLevel(n)) = stratProfile(errorLevel(n)) + 1;
        end  
    end
    chooseTable = ComputeChooseTable(sampleSize);
    threshold = TrainErrorOverfitThreshold(sampleSize, trainSize, [0:sampleSize], eps);
    
    bound = 0;
    for n = 1 : numel(sources)
        bound = bound + HypergeometricTail(sampleSize, trainSize, errorLevel(sources(n)), ....
                                           threshold(errorLevel(sources(n) + 1)), chooseTable);
    end
    for m = 1:sampleSize
        if threshold(m + 1) ~= threshold(m)
            bound = bound + stratProfile(m) * ...
                            HypergeometricSum(sampleSize - 1, trainSize, m - 1, ...
                            threshold(m + 1), threshold(m + 1), chooseTable);
        end
    end
    bound = bound / chooseTable(sampleSize + 1, trainSize + 1);
end