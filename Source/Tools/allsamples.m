function samples = allsamples(n, k)
    len = nchoosek(n,k);
    samples = zeros(len, k);
    
    sample = 1:k;
    samples(1, :) = sample;
    for i = 2:len
        for j = k:-1:1
           lim = n - k + j;
           if (sample(j) < lim)
               break;
           end
        end
        
        
        sample(j) = sample(j) + 1;
        for q = (j+1):k
            sample(q) = sample(q - 1) + 1;
        end        
        
        samples(i, :) = sample;
    end    
end