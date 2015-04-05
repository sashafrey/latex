function [levels] = GetErrorLevels(totalError, numErrorLevels)
    numAlgs = numel(totalError);
    levels = ones(1, numErrorLevels);
	k = 1;
    for m = 1 : numErrorLevels
		while (k <= numAlgs && totalError(k) < m)
			k = k + 1;
        end
		levels(m) = k;
    end
end