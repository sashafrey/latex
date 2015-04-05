function [val] = MonteCarloChainEstimation(chain, L, l, eps, numPart)

    chainLen = size(chain, 2);
    k = L - l;
    val = 0;
 
    m0 = sum(chain == -1); 
    totalError = zeros(1, chainLen + 1);
    totalError(1) = m0;
    for i = 2:chainLen+1
       totalError(i) = totalError(i - 1) + chain(i - 1);  
    end
       
    n = nchoosek(L, l);
    %temp = nchoosek(1: L, l); 
    for j = 1:numPart
        ind = randperm(L);
        ind = ind(1:l);
        ind = ind(ind <= chainLen);
        %ind = temp(j, :);
        minNum = 1;
        obsError = sum(chain(ind) == -1);
        curError = obsError;
        for i = ind
            curError = curError + chain(i);
            if (curError < obsError)
                obsError = curError;
                minNum = i + 1;
            end
        end
        
        val = val +  ((totalError(minNum) - obsError) / k - obsError/l > eps) ;
    end
    
    val = val / numPart;
end