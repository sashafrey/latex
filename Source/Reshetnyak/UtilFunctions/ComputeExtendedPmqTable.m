function [probRMQ] = ComputeExtendedPmqTable(L, ell, eps, maxQ, maxS)
%Рассчитывает Pmq - число выборок, на которых алгоритм, допускающий 
%m ошибок, имеющий q детей переобучится, при условии, что его наименьший
%предок допускает r ошибок 
%maxS - максимально допустимое число ошибок на обучении

	function [res] = G(s, L, ell, m)
        res = 0;
        if (ell < 0 || L < ell || s < 0)
            return
        end
        for i = max(0, m + ell - L) : min(s, min(m, ell))
            res = res + cTable(m + 1, i + 1) * cTable(L - m + 1, ell - i + 1);
        end
    end

    if nargin < 5
        maxS = L;
    end
	cTable = ComputeChooseTable(L);
	probRMQ = zeros(L + 1, L + 1, maxQ + 1);
	
    for r = 0 : L
        for m = r : L
            s = min(maxS, TrainErrorOverfitThreshold(L, ell, m, eps));
            for q = 0:maxQ
                probRMQ(r + 1, m + 1, q + 1) = G(s, L - m + r - q, ell - q, r);
            end
        end
    end
    probRMQ = probRMQ / cTable(L + 1, ell + 1);
	
end