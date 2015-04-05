function [dist, outDegree] = BFS(sourceNumber, graph)
    graphSize = numel(graph);
    dist = inf(graphSize, 1);
    vertexQueue = zeros(graphSize, 1);
    outDegree = zeros(graphSize, 1);
    
    dist(sourceNumber) = 0;
    left = 0;
    right = 1;
    vertexQueue(1) = sourceNumber;
    while (left < right)
        left = left + 1;
        curVertex = vertexQueue(left);
        for v = 1:numel(graph{curVertex})
            if dist( graph{curVertex}(v)) == inf
                right = right + 1;
                vertexQueue(right) = graph{curVertex}(v);
                dist(vertexQueue(right)) = dist(curVertex) + 1;
            end
            if dist(graph{curVertex}(v)) > dist(curVertex)
                outDegree(curVertex) = outDegree(curVertex) + 1;
            end
        end
    end
    
end