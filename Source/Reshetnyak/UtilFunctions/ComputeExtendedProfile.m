function [profile] = ComputeExtendedProfile(graph, errorLevel)
    
    if ~issorted(errorLevel)
        'Error!!! vertices dont sorted by error level.'
    end
    numAlgs = numel(graph);
    vertexQueue = zeros(numAlgs, 1);
    outDegree = zeros(numAlgs, 1);
    minAnsector = inf(numAlgs, 1);
    
     
    for n = 1:numAlgs
        if minAnsector(n) == inf
            minM = errorLevel(n);
            newVertex = 0;
            minAnsector(n) = minM;
            left = 0;
            right = 1;
            vertexQueue(1) = n;
            while left < right
                left = left + 1;
                curVertex = vertexQueue(left);
                for v = 1:numel(graph{curVertex})
                    newVertex = graph{curVertex}(v);
                    if errorLevel(newVertex) > errorLevel(curVertex)
                        if minAnsector(newVertex) == inf
                            right = right + 1;
                            vertexQueue(right) = newVertex;
                            minAnsector(newVertex) = minM;
                        end
                        outDegree(curVertex) = outDegree(curVertex) + 1;
                    end
                end
            end
        end
      
    end
    profile = zeros(max(minAnsector) + 1, max(errorLevel) + 1, max(outDegree) + 1);
    
    for n = 1:numAlgs
        profile(minAnsector(n) + 1, errorLevel(n) + 1, outDegree(n) + 1) = ...
            profile(minAnsector(n) + 1, errorLevel(n) + 1, outDegree(n) + 1) + 1;
    end
    
end