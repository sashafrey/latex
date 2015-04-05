function [familyGraph, levels] = BuildFamilyGraph(algs, numErrorLevels)
%Массив algs - отсортирован по числу ошибок!!!
    [numAlgs L] = size(algs);
    totalError = sum(algs, 2);
    
    if nargin < 2
        numErrorLevels = L + 1;
    end

	levels = GetErrorLevels(totalError, numErrorLevels);
    usedAlgs = levels(numErrorLevels) - 1;
    familyGraph = cell(1, usedAlgs);
	
    for i = 1 : usedAlgs
        if totalError(i) + 1 == numErrorLevels
            break
        end
        for j = levels(totalError(i) + 1) : levels(totalError(i) + 2) - 1
			if sum(algs(j, :) ~= algs(i, :)) == 1
				familyGraph{i}( numel(familyGraph{i} ) + 1) = j;
                familyGraph{j} ( numel(familyGraph{j} ) + 1) = i;
			end            
        end
    end
    
end