function [profile, inferiorObjects, upperNeighbourhood] = ComputeInferiorityProfile(algs, graph)
    %graph и totalError должны быть отсортированы по числу ошибок
    [numAlgs numObjects] = size(algs);
    totalError = sum(algs, 2);
    
    outDegree = zeros(1, numAlgs);
    inferiorObjects = false(numAlgs, numObjects);
    if nargout > 2
        upperNeighbourhood = cell(1, numAlgs);
    end
    
    for n = 1:numAlgs
        for v = graph{n}
            if totalError(v) == totalError(n) + 1
                outDegree(n) = outDegree(n) + 1;
            elseif totalError(v) < totalError(n)
                inferiorObjects(n, :) = inferiorObjects(n, :) | inferiorObjects(v, :);
                inferiorObjects(n, algs(n, :) ~= algs(v, :)) = true;
            end
        end
        if nargout > 2
            upperNeighbourhood{n} = zeros(1, outDegree(n));
            pos = 0;
            for v = graph{n}
                if totalError(v) == totalError(n) + 1
                    pos = pos + 1;
                    upperNeighbourhood{n}(pos) = find(algs(n, :) ~= algs(v, :));
                end
            end
        end
    end
    
    inferiority = sum(inferiorObjects, 2);
    profile = zeros(max(totalError) + 1, max(totalError) + 1, max(outDegree) + 1);
    for n = 1:numAlgs
        profile(totalError(n) - inferiority(n) + 1, totalError(n) + 1, outDegree(n) + 1) = ...
            profile(totalError(n) - inferiority(n) + 1, totalError(n) + 1, outDegree(n) + 1) + 1;
    end
         
end