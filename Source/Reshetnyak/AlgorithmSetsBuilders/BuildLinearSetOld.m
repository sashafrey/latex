
%Строит по двумерной выборке sample и её вектору классификации 
% sampleClasses всевозможные разделяющик прямые
function [algs] = BuildLinearSetOld(sample, sampleClasses)

% 	function AddAlgorithm(a)
% 		%isNew = true;
% 		for q =  1:numAlgs
% 			if (a == algs(q, :))
% 				return ;
% 			end
% 		end
% 		
% 		numAlgs = numAlgs + 1;
% 		algs(numAlgs, :) = a;
% 	end

   
    [sampleSize dim] = size(sample);
    algs = false(4 * sampleSize * (sampleSize - 1), sampleSize);
    sampleClasses(sampleClasses == 0) = -1;
    num = 0;
    for i = 1:sampleSize
        for j = i+1:sampleSize
            %Перебираем все пары точек выборки
            if (i ~= j)
               %Вычисляем вектор классификации для прямой, проходящей через i и j точку выборки 
               a = false(1, sampleSize);
               for h = 1:sampleSize
                   %Работает только при dim = 2
                   v = (sample(h, 1) - sample(i, 1) ) * (sample(j, 2) - sample(i, 2) ) - (sample(h, 2) - sample(i, 2) ) * (sample(j, 1) - sample(i, 1) );
                   a(h) = (v * sampleClasses(h) <  0);
               end
               
            
               %"Пошевелим" прямую
               %Получаем 8 вариантов:
               %a(i) = 1, a(j) = 1
               %a(i) = 1, a(j) = 0
               %a(i) = 0, a(j) = 1
               %a(j) = 1, a(i) = 0;
               
			   num = num + 1;
               algs(num, :) = a;
			   %AddAlgorithm(a);
			   
			   num = num + 1;
			   algs(num, :) = ~a;
			   %AddAlgorithm(~a);
			   
			   num = num + 1;
			   a(i) = ~a(i);
               algs(num, :) = a;
			   %AddAlgorithm(a);
			  
			   num = num + 1;
			   algs(num, :) = ~a;
			   %AddAlgorithm(~a);
              
			   num = num + 1;
               a(j) = ~a(j);
               algs(num, :) = a;
			   %AddAlgorithm(a);
			   
			   num = num + 1;
			   algs(num, :) = ~a;
			   %AddAlgorithm(~a);			   
			   
               num = num + 1;
               a(i) = ~a(i);
               algs(num, :) = a;
			   %AddAlgorithm(a);
			   
			   num = num + 1;
			   algs(num, :) = ~a;
			   %AddAlgorithm(~a);
               
            end
        end
    end
    %Удаляем одинаковые алгоритмы
    algs = sortrows(algs);
	n = 1;
	for s = 2:size(algs, 1)
		if all(algs(s, :) == algs(s - 1, :))
			continue;
		end
		n = n + 1;
		algs(n, :) = algs(s, :);
    end
    algs = algs(1:n, :);
    terr = sum(algs, 2);
    [terr, ind] = sort(terr);
    algs = algs(ind, :);
    
    %algs(2: size(algs, 1), :) = algs(randperm(size(algs, 1) - 1) + 1, :);
    
end