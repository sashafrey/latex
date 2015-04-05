function [val] = ExactPessimisticErm(A, l, eps)
    [numAlgs L] = size(A);
    
    k = L - l;
    val = 0;
    ind = nchoosek(1: L, l); 
    totalError =  sum(A , 2)';
    val = zeros(1, numAlgs);
    
    for j = 1:nchoosek(L, l)
    
        observedError = sum(A(:, ind(j, :) ), 2)';
        minObservedError = zeros(1, numAlgs);
        maxTotalError = zeros(1, numAlgs);
        minObservedError(1) = observedError(1);
        maxTotalError(1) = totalError(1);
        for i = 2:numAlgs
            minObservedError(i) = observedError(i);
            maxTotalError(i) = totalError(i);
            if (minObservedError(i) > minObservedError(i - 1) )
                 minObservedError(i) = minObservedError(i - 1);
                 maxTotalError(i) = maxTotalError(i - 1);
            end
            if ( (minObservedError(i) == minObservedError(i - 1) ) & ( maxTotalError(i) < maxTotalError(i - 1) ) )
                 maxTotalError(i) = maxTotalError(i - 1);
            end
        end
        
        val = val +  ((maxTotalError - minObservedError) / k - minObservedError/l >= eps) ;
    end
    
    val = val / nchoosek(L, l);
    
end