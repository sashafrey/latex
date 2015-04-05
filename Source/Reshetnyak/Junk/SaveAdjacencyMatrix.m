function SaveAdjacencyMatrix(filename, algs)
    [numAlgs L] = size(algs);

    adj = zeros(numAlgs, numAlgs);
    for i = 1:numAlgs
        for j = i + 1:numAlgs
            if ( sum( abs(algs(i, :) - algs(j, :) ) ) == 1)
                adj(i, j) = 1;
                %adj(j, i) = 1;
            end
        end
    end
    
    
    f = fopen(filename, 'w');
    fprintf(f, 'PIG:0\nLinearFamily\n', numAlgs)
     for i = 1:numAlgs
        for j = i + 1:numAlgs
            if (adj(i, j) )
                fprintf(f, '%d %d\n', i, j );
            end
        end
     end
    fprintf(f, '0 0\n');
    fclose(f);
end