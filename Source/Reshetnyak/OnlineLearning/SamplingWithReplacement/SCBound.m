function [overfitProb] = SCBound(algs, graph, objectsDistr)
    [maxTime numObjects] = size(objectsDistr);
    if numObjects ~= size(algs, 2)
        'ERROR!!! algs and objectDistr sizes does not match'
        return
    end
    
    numAlgs = size(algs, 1);
    [profile, inferiorObjects, un] = ComputeInferiorityProfile(algs, graph);
    inferiorObjects = ~inferiorObjects;
    maxUNSize = max(cellfun('size', un, 2));
    overfitProb = zeros(1, maxTime - 1);
    %algChooseProb = ones(maxTime, numAlgs);
    algChooseProb = zeros(maxTime, numAlgs, maxUNSize + 1);
    algChooseProb(:, :, 1) = 1;
    for t = 2 : maxTime
        for n = 1 : numAlgs
            bestAlgProb = sum(objectsDistr(t - 1, inferiorObjects(n, :)));
            upperProb = sum(objectsDistr(t - 1, un{n}));
            errorProb = sum(objectsDistr(t, algs(n, :)));
            for s = 1 : numel(un{n}) + 1
                algChooseProb(t, n, s) = algChooseProb(t - 1, n, s) * bestAlgProb;
                if s > 1
                    algChooseProb(t, n, s) = algChooseProb(t, n, s) + bestAlgProb * upperProb ...
                        * (algChooseProb(t - 1, n, s - 1) - algChooseProb(t - 1, n, s));
                end
                %overfitProb(t - 1) = overfitProb(t - 1) + algChooseProb(t, n) * sum(objectsDistr(t, algs(n, :)));
                %algChooseProb(t, n) = algChooseProb(t - 1, n)  + log(sum(objectsDistr(t - 1, inferiorObjects(n, :))));
                
            end
            overfitProb(t - 1) = overfitProb(t - 1) + algChooseProb(t, n, numel(un{n}) + 1) * errorProb;
        end
    end
end