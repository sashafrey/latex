function [res] = GetMaxEdgeLength(algs, graph)
    totalError = sum(algs, 2);
    res = 0;
    for n = 1 : numel(graph)
        for v = graph{n}
            res = max(res, totalError(v) - totalError(n));
        end
    end
end