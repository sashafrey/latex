%Функция выдает вектор val с длиной, равной числу алгоритмов в 
%семействе, где val(n) - оценка связности-расслоения для вероятности
%переобучения семейства из первых n алгоритмов
%А - семейство алгоритмов
%familyGraph - граф семейства алгоритмов
%l - длина обучающей выборки
%eps - требуемая точность предсказания
%flag - параметр алгоритма
%Если flag = 'simple'
function [val] = ERMFunctional(A, familyGraph, l, eps, flag)
   
	if (nargin == 4)
		flag = 'simple';
	end
    [numAlgs L] = size(A);
    k = L - l;
    
    %A(1, :) = zeros(1, L);
    totalError = sum(A, 2);
    
	m0 = 0;
	levels = cell(1, L + 1);
	for m = 0:L
		levels{m + 1} = find(totalError == m);
		if (numel(levels{m + 1}) == 0) 
			m0 = m + 1;
		end
	end
        
    branching = zeros(1, L + 1); 
    childCount = zeros(1, numAlgs);
	bestCount = zeros(1, L + 1);
   
    val = zeros(1, numAlgs);
	
	maxQ = 0;
	for n = 1:numAlgs
		maxQ = max(maxQ, numel(familyGraph{n}));
	end
	if ( strcmp(flag,'simple') | strcmp(flag,'sphere') | strcmp(flag, 'simple_connect') )
		countTable = ComputePmqTable(L, l, eps, m0, maxQ);
	end
	if ( strcmp(flag,'rival') )
		countTable = ComputeRivalTable(L, l, eps, m0, maxQ);
	end
	
	
    val(1) = countTable(m0 + 1, 1);
    D = zeros(1, L + 1);
    D(m0 + 1) = 1;
	
	D1 = zeros(L + 1, maxQ + 1);
	D1(m0 + 1, 1) = 1; 
	
	branching = zeros(1, L + 1);
    for n = 2:numAlgs
        
        lev = totalError(n); %число ошибок n-го алгоритма
		D(lev + 1) = D(lev + 1) + 1;
        
        edges = familyGraph{n}';
        childCount(n) = sum( (edges < n) .* (totalError(edges) == lev  + 1) );
		D1(lev + 1, childCount(n) + 1) = D1(lev + 1, childCount(n) + 1) + 1;
        
		parents = edges( (edges < n) & (totalError(edges) == lev  - 1) );
		
		D1(lev, childCount(parents) + 1) = D1(lev, childCount(parents) + 1) - 1;
        childCount(parents) =  childCount(parents) + 1;
		D1(lev, childCount(parents) + 1) = D1(lev, childCount(parents) + 1) + 1;
		
       	%maxQ
		%branching
		%childCount
		if  (strcmp(flag, 'simple') | strcmp(flag,'rival') )
			
			branching(lev + 1) = min( childCount( levels{lev + 1}(levels{lev + 1} <= n) ) );
			branching(lev) = min( childCount( levels{lev}(levels{lev} <= n) ) );
			for m = m0:L
				val(n) = val(n) +  D(m + 1) * countTable(m + 1, branching(m + 1)  + 1);
			end
		end
		if (strcmp(flag, 'simple_connect') )
			for m = m0:L
				for q = 0:maxQ
					val(n) = val(n) +  D1(m + 1, q + 1) * countTable(m + 1, q + 1);
				end
			end
		end
			
    end
    
    val = val(numAlgs)


end
   