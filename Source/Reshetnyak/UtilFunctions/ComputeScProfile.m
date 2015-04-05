function [profile, isOneSource] = ComputeScProfile(sourceNumber, graph)
    [dist, outDegree] = BFS(sourceNumber, graph);
    profile = zeros(max(dist) + 1, max(outDegree) + 1);
    numAlgs = numel(graph);
    isOneSource = true;
    for n = 1:numAlgs
        if n ~= sourceNumber && outDegree(n) == numel(graph{n})
          isOneSource = false;
        end
        profile( dist(n) + 1, outDegree(n) + 1 ) = profile( dist(n) + 1, outDegree(n) + 1) + 1;
    end
end