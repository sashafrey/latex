function [res] = ComputeGeneralizedGraphBoundTest(algs, graph, eps, trainSize, maxEdgeLength)

    function [distr] = BtunProb(n, trainSize)
        
        function [pos] = GetPos(bitmask) 
            pos = 0;
            for i = 1 : numel(bitmask)
                pos = 2 * pos + bitmask(i);
            end
            pos = pos + 1;
        end
        
        function [mask] = GetMask(pos, maskSize)
            pos = pos - 1;
            mask = false(1, maskSize);
            for i = maskSize : -1 : 1
                mask(i) = mod(pos, 2);
                pos = floor(pos / 2); 
            end
        end
        
        function [edges, newIndices] = ConstructEdges(n, longEdges)
            upperObjects = false(1, sampleSize);
            for v = graph{n}
                if (totalError(v) > totalError(n) + 1) && (totalError(v) <= totalError(n) + maxEdgeLength)
                    upperObjects = upperObjects | xor(algs(n, :), algs(v, :));
                end
            end
            newIndices = cumsum(upperObjects);
            edges = false(longEdges, sum(upperObjects));
            cur = 0;
            for v = graph{n}
                if (totalError(v) > totalError(n) + 1) && (totalError(v) <= totalError(n) + maxEdgeLength)
                    cur = cur + 1;
                    edges(cur, newIndices(xor(algs(n, :), algs(v, :)))) = true; 
                end
            end
        end
        
        function [simpleEdges, longEdges] = CountEdges(n)
            simpleEdges = 0;
            longEdges = 0;
            for v = graph{n}
                if totalError(v) > totalError(n)
                    if totalError(v) == totalError(n) + 1
                        simpleEdges = simpleEdges + 1;
                    elseif totalError(v) <= totalError(n) + maxEdgeLength
                        longEdges = longEdges + 1;
                    end
                end
            end
        end
        
        L = sampleSize - totalError(n);
        [simpleEdges longEdges] = CountEdges(n);
        
        if longEdges == 0
            distr = zeros(1, simpleEdges + 1);
            distr(simpleEdges + 1) = 1;
            return
        end
        edges = ConstructEdges(n, longEdges);
        edges = edges';
        numObjects = size(edges, 1);
        posToMask = false(2^longEdges, longEdges);
        for n = 1 : 2^longEdges
            posToMask(n, :) = GetMask(n, longEdges);
        end
        dp = zeros(numObjects + 1, 2^longEdges);
        dp(1, 1) = 1;
        for ii = 1 : numObjects
            for j = ii:-1:1
                for h = 1 : size(dp, 2)
                    if dp(j, h) == 0
                        continue;
                    end
                    t = GetPos(posToMask(h, :) | edges(ii, :)); 
                    dp(j + 1, t) = dp(j + 1, t) + dp(j, h);
                end
            end
        end
        distr = zeros(1, simpleEdges + numObjects + 1);
        for ii = 1 : numObjects + 1
            distr(simpleEdges + ii) = dp(ii, 2^longEdges);
        end
    end

    [numAlgs sampleSize] = size(algs);
    if nargin < 5
        maxEdgeLength = sampleSize
    end
    chooseTable = ComputeChooseTable(sampleSize + 1);
    totalError = sum(algs, 2);
    [waste, inferiorObjects] = ComputeInferiorityProfile(algs, graph);
    inferiority = sum(inferiorObjects, 2);
    threshold  = TrainErrorOverfitThreshold(sampleSize, trainSize, 0 : max(totalError), eps);
    overfitProb = zeros(1, numAlgs);
    minError = min(totalError);
    for n = 1 : numAlgs
        m = totalError(n);
        distr = BtunProb(n);
        unSize = numel(distr) - 1;
        for s = max(0, trainSize + m - sampleSize) : min(totalError, min(threshold(totalError(n) + 1), m - inferiority(n)))
            for q = 0 : min(unSize, trainSize - s)
                overfitProb(n) = overfitProb(n) + chooseTable(m - inferiority(n) + 1, s + 1) * ...
                    distr(q + 1) * chooseTable(sampleSize - m - unSize + 1, trainSize - s - q + 1);
            end
        end
    end
    res = sum(sort(overfitProb)) / chooseTable(sampleSize + 1, trainSize + 1);
end