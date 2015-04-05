%Оставляет среди всех алгоритмов только уникальные и сортирует их по числу
%ошибок
function [uniqueAlgs] = GetUniqueAlgs(algs)
    algs = sortrows(algs);
    n = 1;
	for s = 2:size(algs, 1)
		if all(algs(s, :) == algs(s - 1, :))
			continue;
		end
		n = n + 1;
		algs(n, :) = algs(s, :);
    end
    uniqueAlgs = algs(1:n, :);
    terr = sum(uniqueAlgs, 2);
    [terr, ind] = sort(terr);
    uniqueAlgs = uniqueAlgs(ind, :);
end