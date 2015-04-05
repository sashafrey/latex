function [probMQ] = ComputeAlgorithmChoiceProbTable(L, ell, m0, maxQ)
%–ассчитывает матождидание числа ошибок на контроле алгоритма, допускающего
%m ошибок и имеющиего q исход€щих рЄбер, когда он может быть выбран пессимистическим ERM  

	function [res] = F(L, ell, m0, m)
        res = 0;
        if (ell < 0 || L < ell)
            return
        end
        for s = max(0, m0 + ell - L) : min(m0, ell) 
            res = res +  (m - s) * cTable(m0 + 1, s + 1)* cTable(L + 1, ell - s + 1);
        end
	end

	cTable = ComputeChooseTable(L);
	probMQ = zeros(L + 1, maxQ + 1);
	
	for m = m0:L
		for q = 0:maxQ
			probMQ(m + 1, q + 1) = F(L - m - q, ell - q, m0, m);
		end
    end
    probMQ = probMQ / cTable(L + 1, ell + 1);
	
end