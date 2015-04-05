function OutputGraphForDrawer(algs)
    errorCount = sum(algs, 2);
    [errorCount, ind] = sort(errorCount);
    algs = algs(ind, :);
    graph = BuildFamilyGraph(algs);
    edges = zeros(3, 1)
    cnt = 0;
    numInLayer = 0;
    for n = 1:numel(graph)
        for v = 1:numel(graph{n})
            if graph{n}{v} > n
                cnt = cnt + 1;
                edges(cnt, :) = [ n]
            end
    end
end