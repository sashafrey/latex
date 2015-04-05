function [distr] = GetEdgesDistribution(algs, graph)
    totalError = sum(algs, 2);
    distr = zeros(1, GetMaxEdgeLength(algs, graph));
    for n = 1 : numel(graph)
        for v = graph{n}
            distr(totalError(v) - totalError(n)) = distr(totalError(v) - totalError(n)) + 1;
        end
    end
end