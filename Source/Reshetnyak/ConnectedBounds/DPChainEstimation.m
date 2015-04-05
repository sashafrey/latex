function[val] = DPChainEstimation(chain, L, l, eps)

    chainLen = size(chain, 2);
    nGood = L - chainLen;
    m0 = sum(chain == -1);
    k = L - l;
    val = 0;
 
    totalError = zeros(1, chainLen + 1);
    totalError(1) = m0;
    for i = 2:chainLen+1
       totalError(i) = totalError(i - 1) + chain(i - 1);  
    end
    
    plot([1:chainLen+1], totalError, 'g');
    
    lb = max(floor(l/L * (totalError - k * eps) ) + 1, zeros(1, chainLen + 1) ) + 1;
    rb = min(totalError, l) + 1;
    
    for i = 2:chainLen+1
        if (chain(i - 1) == 1 && lb(i) == 1)
            lb(i) = 2;
        end
        if (chain(i - 1) == -1 && rb(i) == l + 1)
            rb(i) = l;
        end      
    end
    
    cTable = zeros(1, nGood + 1);
    for i = 0:nGood
        cTable(i + 1) = nchoosek(nGood, i);
    end
    
        
    for stPos = lb(1):rb(1)
        dp = zeros(l + 1, l + 1);
        dp(1, stPos) = 1;
        for pos = 2:chainLen+1
          
            f = chain(pos - 1);
            for r = min(pos, l + 1):-1:2
                for s = lb(pos):rb(pos)
                    dp(r, s) = dp(r, s) + dp(r - 1, s - f);
                end
            end
           
        end
        for l1 = max(stPos - 1, l - nGood) : l
            val = val + cTable(l - l1 + 1) * dp(l1 + 1, l1 - stPos + 2);
        end
    end
       
    val = 1.0 - val /nchoosek(L, l);
end
