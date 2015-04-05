function [probMQ] = ComputeRivalTable(L, l, eps, m0, maxQ)
	%Рассчитывает probMQ - число выборок, на которых алгоритм, допускающий 
%m ошибок и имеющий q детей переобучится, при условии, что лучший алгоритм
%в семейсве допускает допускает m0 ошибок 

	function [res] = ComputeRivalProb(m1, q);
		smax = floor( l/ L * (m1 - (L - l) * eps) );
		res = 0;
		for s1 = 0:smax
			for s0 = s1:m0
				res = res + cTable(m0 + 1, s0 + 1) * cTable(m1 + 1, s1 + 1) * cTable(L - m0 - m1 - q + 1, l - s0 - s1 - q);
			end
		end
	end

	cTable = ComputeChooseTable(L);
	probMQ = zeros(L + 1, maxQ + 1);
	
	for m = m0:L
		for q = 0:maxQ
			probMQ(m + 1, q + 1) = ComputeRivalProb(m, q);
		end
	end
end