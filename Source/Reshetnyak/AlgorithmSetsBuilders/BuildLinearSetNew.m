
%Строит по n-мерной выборке sample и её вектору классификации 
% sampleClasses всевозможные разделяющик прямые
function [uniqAlgs] = BuildLinearSetSimple(sample, sampleClasses)

    [sampleSize dim] = size(sample);
    algs = false(4 * sampleSize * (sampleSize - 1), sampleSize);
    sampleClasses(sampleClasses == 0) = -1;
    numAlgs = 

    
    %Удаляем одинаковые алгоритмы
    algs = sortrows(algs);
    uniqAlgs = false(sampleSize * (sampleSize - 1) + 2, sampleSize);
	n = 1;
	uniqAlgs(1, :) =  algs(1, :);
	for i = 2:size(algs, 1)
		if (algs(i, :) == algs(i-1, :) )
			continue;
		end
		n = n + 1;
		uniqAlgs(n, :) = algs(i, :);
		
	end
   
    terr = sum(uniqAlgs, 2);
    [terr, ind] = sort(terr);
    uniqAlgs = uniqAlgs(ind, :); 
end