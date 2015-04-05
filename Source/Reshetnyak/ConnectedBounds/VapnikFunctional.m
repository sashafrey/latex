function[val] = VapnikFunctional(A, l, eps)

    [numAlgs L] = size(A);
    k = L - l;
    totalError = sum(A, 2);
   
    mvals = [1:L]; 
    svals = floor( l/L * (mvals - eps * k) );
    ind = find(svals >= 0);
    coeff =  hygecdf(svals(ind), L , mvals(ind), l );
    shift = ind(1) - 1;
  
    val = zeros(1, numAlgs);
    D = zeros(1, L);
    for n = 1:numAlgs
        if (totalError(n) - shift > 0) 
            val(n) = coeff(totalError(n) - shift);
        end
        if (n > 1) 
            val(n) = val(n) + val(n - 1);
        end
        
        val(n) = n * max(coeff);
    end
    val = val(numAlgs);
    
end