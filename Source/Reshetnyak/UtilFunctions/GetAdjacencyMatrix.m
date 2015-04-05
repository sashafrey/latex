function [adj] =  GetAdjacencyMatrix(algs)
   
	
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
end