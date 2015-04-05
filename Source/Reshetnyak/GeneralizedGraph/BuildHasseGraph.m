function [familyGraph, levels] = BuildHasseGraph(algs, numErrorLevels, maxEdgeWeight)
%Массив algs - отсортирован по числу ошибок!!!
    [numAlgs L] = size(algs);
    totalError = sum(algs, 2);
    
    if nargin < 3
        maxEdgeWeight = L + 1;
    end
    if nargin < 2
        numErrorLevels = L + 1;
    end

	levels = GetErrorLevels(totalError, numErrorLevels);
    usedAlgs  = levels(numErrorLevels) - 1;
    familyGraph = cell(1, usedAlgs); 
	
    connectedVertices = false(usedAlgs, usedAlgs);
    for i = usedAlgs - 1 : -1: 1   
        for j = i + 1 : levels(min(totalError(i) + maxEdgeWeight, numErrorLevels)) - 1
            if connectedVertices(i, j) == false
                if all(algs(j, :) >= algs(i, :))
                    familyGraph{i}( numel(familyGraph{i} ) + 1) = j;
                    familyGraph{j}( numel(familyGraph{j} ) + 1) = i;
                    connectedVertices(i, :) = connectedVertices(i, :) | connectedVertices(j, :);
                    connectedVertices(i, j) = true;
                end
			end            
        end
    end
    
end