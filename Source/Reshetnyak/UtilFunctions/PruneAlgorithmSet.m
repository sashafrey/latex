function [res] = PruneAlgorithmSet(algs, l);

	totalError = sum(algs, 2);
    [terr, ind] = sort(totalError);
    algs = algs(ind, :);
	
	[numAlgs L] = size(algs);
	f = nchoosek(L, l);
	count = zeros(1, L + 1)
	for i = 1:L-l
		count(i) = nchoosek(L - i + 1, l);
	end
	for n = 1:numAlgs
		if (count(totalError(n) + 1) * 1e7 < f)
			break;
		end
	end
	res = algs(1:n-1, :);
end