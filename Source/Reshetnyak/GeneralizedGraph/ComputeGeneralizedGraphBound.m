function [res] = ComputeGeneralizedGraphBound(algs, graph, eps, trainSize)

    function [res] = BtunProb(n, trainSize)
        
        function [res] = ComputeBtunSimpleInclusionExclusion(depth, sign, usedObjects)
            if depth == longEdges + 1
                res = sign * (chooseTable(L - sum(usedObjects)+ 1, trainSize + 1) ...
                        - chooseTable(L - sum(usedObjects) - simpleEdges + 1, trainSize - simpleEdges + 1));
                return
            end
            res = ComputeBtunSimpleInclusionExclusion(depth + 1, sign, usedObjects);
            usedObjects = usedObjects | edges(depth, :);
            res = res + ComputeBtunSimpleInclusionExclusion(depth + 1, -sign, usedObjects);
            res = res + sign * chooseTable(L - sum(usedObjects)+ 1, trainSize + 1);
        end
        
        function [edges] = ConstructEdges(n, longEdges)
            upperObjects = false(1, sampleSize);
            for v = graph{n}
                if totalError(v) > totalError(n) + 1
                    upperObjects = upperObjects | xor(algs(n, :), algs(v, :));
                end
            end
            newIndices = cumsum(upperObjects);
            edges = false(longEdges, sum(upperObjects));
            cur = 0;
            for v = graph{n}
                if totalError(v) > totalError(n) + 1
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
                    else
                        longEdges = longEdges + 1;
                    end
                end
            end
        end
        
        L = sampleSize - totalError(n);
        [simpleEdges longEdges] = CountEdges(n);
        
        if longEdges == 0
            if simpleEdges > trainSize
                res = 0;
            else
                res = chooseTable(L - simpleEdges + 1, trainSize - simpleEdges + 1) / chooseTable(L + 1, trainSize + 1);
            end
            return
        end
        edges = ConstructEdges(n, longEdges);
        res = 1  - ComputeBtunSimpleInclusionExclusion(1, 1, false(1, size(edges, 2))) / chooseTable(L + 1, trainSize + 1);
    end

    [numAlgs sampleSize] = size(algs);
    chooseTable = ComputeChooseTable(sampleSize + 1);
    totalError = sum(algs, 2);
    [waste, inferiorObjects] = ComputeInferiorityProfile(algs, graph);
    inferiority = sum(inferiorObjects, 2);
    threshold  = TrainErrorOverfitThreshold(sampleSize, trainSize, 0 : max(totalError), eps);
    overfitProb = zeros(1, numAlgs);
    for n = 1 : numAlgs
        m = totalError(n);
        for s = max(0, trainSize + m - sampleSize) : min(threshold(totalError(n) + 1), m - inferiority(n))
            overfitProb(n) = overfitProb(n) + chooseTable(m - inferiority(n) + 1, s + 1) * ...
                  chooseTable(sampleSize - m + 1, trainSize - s + 1) * ...
                  BtunProb(n, trainSize - s);
        end
    end
    res = sum(sort(overfitProb)) / chooseTable(sampleSize + 1, trainSize + 1);
end