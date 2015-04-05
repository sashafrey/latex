function hashVal = vectorHash(vector)
    % this function turns out to be crutial for performance.
    % use it only for vectors with length <=5 and values <= 100.

    hashVal = 0;
    prime = 100;
    vector(isnan(vector)) = 99;
    for i = 1:length(vector)
        hashVal = hashVal * prime + vector(i);
    end
    
    return;    
 
    % this would be a proper hash, but it is too slow:
    prime = uint64(997);
    modulus = uint64(2147483647); % this is prime number.
    
    assert(all(vector < (prime - 1) | isnan(vector)));    
    
    hashVal = uint64(1);
    
    for i = 1:length(vector)
        curVal = vector(i);
        if (isnan(curVal))
            curVal = 996;
        end
        hashVal = mod(hashVal * prime + uint64(curVal), modulus) ;
    end
end