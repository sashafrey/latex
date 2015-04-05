function [val] = RecurrentBoundCalculation(algs, l, eps)

	function [res] = G(s, L, l, m)
		res = 0;
		if (l < 0 || L < l || s < 0)
			return
		end
        for i = max(0, m + l - L) : min(s, m)
			res = res + cTable(m + 1, i + 1) * cTable(L - m + 1, l - i + 1);
		end
	end

	function UpdateInformation(num)
		
		t = size(algsInf{num}, 1);
		for j = 1:t
			if (algsInf{num}{j, 1} == 0)
				continue;
			end
             
			if ( numel( intersect( algsInf{num}{j, 2}, curErrors) ) > 0 )
				continue
			end
			b = setdiff(curErrors,  algsInf{num}{j,3});
			if (numel(b) == 1)
				algsInf{num}{j, 2}(  numel(algsInf{num}{j,2}) + 1) = b(1);
			end
			if (numel(b) > 1)
				last = size(algsInf{num}, 1) + 1;
				
				algsInf{num}{last,1} =  -algsInf{num}{j,1};
                algsInf{num}{last,2} =  algsInf{num}{j,2};
				algsInf{num}{last,3} = union( algsInf{num}{j, 3}, curErrors);  
			end
			if (numel(b) == 0)
				algsInf{num}{j, 1} = 0;
			end
		end
	end

	function DeleteInvalidIndexSets(num)
		
		t = size(algsInf{num}, 1);
		pos = 1;
		for j = 1:t
			if ( (algsInf{num}{j,1} ~= 0) & (numel(algsInf{num}{j,2}) <= l) & ...
					(numel(algsInf{num}{j, 3}) <= k) ) 
				if (pos ~= j)
					algsInf{num}(pos, :) = algsInf{num}(j, :);

				end
				pos = pos + 1;
			end
		end
		
		if (pos <= t)
			algsInf{num}(pos : t, :) = [];
		end
	end

	function Evaluate(n)
		
		val(n) = 0;
		for ii = 1:n
			t = size(algsInf{ii}, 1);
			for j = 1:t
% 				n
% 				ii 
% 				j
% 				'first'
% 				algsInf{ii}{1}{j}
% 				'second'
% 				algsInf{ii}{2}{j}
% 				'third'
% 				algsInf{ii}{3}{j}
				val(n) = val(n) + algsInf{ii}{j,1} * G(svals(ii) - sum( algs(ii, algsInf{ii}{j,2}) ), ...
					L - numel(algsInf{ii}{j,2}) - numel(algsInf{ii}{j,3}), l - numel(algsInf{ii}{j,2}),  ...
					totalError(ii) -  sum( algs(ii, algsInf{ii}{j,2}) ) - sum( algs(ii, algsInf{ii}{j,3}) ) );
			end 
		end

		%Вычисление вероятности переобучения
		val(n) = val(n)/ cTable(L + 1, l + 1);
		
	end

	[numAlgs L] = size(algs);
    k = L - l;
    
    totalError = sum(algs, 2);
    [totalError, ind] = sort(totalError);
    algs = algs(ind, :);
    
    algsInf = cell(numAlgs, 1);
    
    val = zeros(1, numAlgs);
    algsInf{1} =  { 1, [] , [] } ;
    val(1) = 0;
    
    cTable = zeros(L + 1, L + 1);
    for i = 1:L+1
        cTable(i, 1) = 1;
        for j = 2:i 
            cTable(i, j) = cTable(i - 1, j - 1) + cTable(i - 1, j);
        end
    end
    
    svals = floor(l/L * ( totalError - eps * k));
    
    for n = 2:numAlgs
        
		if (totalError(n) > k) 
			break;
		end
		%Коррекция индексных множеств
		curErrors = find(algs(n, :) == 1);
		algsInf{n} =  { 1, [], curErrors } ;
		for i = 1:n-1
			UpdateInformation(i);
			DeleteInvalidIndexSets(i);
		end
      
		n
		Evaluate(n);
		
	end
	val(n:numAlgs) = val(n - 1);
    
end