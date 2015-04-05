function CheckHasseGraph(totalError, graph)
    numAlgs = numel(graph)
    adjMatrix = false(numAlgs, numAlgs);
    for n = 1 : numAlgs
        for v = graph{n}
            adjMatrix(n, v) = true;
        end
    end
    good = true;
    for n = 1 : numAlgs
        for i = 1 : numel(graph{n})
           for j = i + 1 : numel(graph{n})
               if adjMatrix(graph{n}(i), graph{n}(j))
                   n
                   graph{n}{i}
                   graph{n}{j}
               end
           end
        end
    end
    if ~good
        'Invalid hasse graph'
    end
end