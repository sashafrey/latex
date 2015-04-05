function [A] = BuildAlgorithmSet(L, m0, numAlgs,  branchFactor )
    A = zeros(numAlgs, L);
    A(1, :) = [ ones(1, m0) zeros(1, L - m0) ];
    
    
    cur = 1;
    count = 1;
    while (count < numAlgs && cur <= count)
      
        curDeg = branchFactor;
        for i = 1:cur
            if ( sum( abs(A(cur, :) - A(i, :) ) ) == 1) 
                curDeg = curDeg - 1;
            end
        end
        
        curDeg
        t = randsample( find( A(cur, :) == 0), curDeg );
        for i = 1:size(t, 2)
            temp = A(cur, :);
            temp(t(i)) = 1;
            flag = true;
        
            for i = 1:count
                if (temp == A(i, :) )
                    flag = false;
                end
            end
        
            if (~flag) 
                continue;
            end
        
            count = count + 1;
            A(count, :) = temp; 
        end
        cur = cur + 1;
end