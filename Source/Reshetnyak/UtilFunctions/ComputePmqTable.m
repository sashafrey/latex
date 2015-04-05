function [probMQ] = ComputePmqTable(L, ell, eps, m0, maxQ)
%Рассчитывает Pmq - число выборок, на которых алгоритм, допускающий 
%m ошибок и имеющий q детей переобучится, при условии, что его наименьший
%предок допускает m0 ошибок 

	function [res] = G(s, L, ell, m)
        res = 0;
        if (ell < 0 || L < ell || s < 0 || s > ell)
            return
        end
        for i = max(0, m + ell - L) : min(s, m)
            res = res + cTable(m + 1, i + 1) * cTable(L - m + 1, ell - i + 1);
        end
	end

	cTable = ComputeChooseTable(L);
	probMQ = zeros(L + 1, maxQ + 1);
	
    for m = m0:L
		s = TrainErrorOverfitThreshold(L, ell, m, eps);
        for q = 0:maxQ
            probMQ(m + 1, q + 1) = G(s, L - m + m0 - q, ell - q, m0);
        end
    end
    probMQ = probMQ / cTable(L + 1, ell + 1);
	
end