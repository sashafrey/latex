function [updatedAlgs] = AddPseudoAlgorithms(algs, numErrorLevels, maxEdgeWeight)
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
    newAlgs = false(size(algs));
    newAlgsCount = 0;
    for i = 1 : usedAlgs   
        for j = i + 1 : levels(min(totalError(i) + maxEdgeWeight, numErrorLevels)) - 1
            if (sum(xor(algs(i, :), algs(j, :))) <= maxEdgeWeight) && any(algs(j, :) <= algs(i, :)) 
                newAlgsCount = newAlgsCount + 1;
                newAlgs(newAlgsCount, :) = algs(i, :) | algs(j, :);
            end
        end
    end
    updatedAlgs = GetUniqueAlgs([algs; newAlgs(1 : newAlgsCount, :)]);
end