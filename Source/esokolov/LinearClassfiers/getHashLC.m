function hash = getHashLC(alg)
    N = length(alg.errVect);

    modulus = 2 ^ 32;
    
    exp_base = 31;
    
    %coefs = 31 .^ (N:-1:1);
    coefs = zeros(N, 1);
    for i = 1:length(coefs)
        coefs(i) = fastExponentiationMod(exp_base, N - i + 1, modulus);
    end
    
    hash = sum(coefs .* alg.errVect);
    hash = mod(hash, mod_const);
end
