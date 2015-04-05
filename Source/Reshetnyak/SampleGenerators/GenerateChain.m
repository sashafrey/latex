function [res] = GenerateChain(L, p)
    res = zeros(1, L);
   
    for i=1:L
        if (2*i == L) 
            p = 1 - p;
        end
            
        if ( rand() > p ) 
            res(i) = 1;
        else
             res(i) = -1;
        end
       
    end
end