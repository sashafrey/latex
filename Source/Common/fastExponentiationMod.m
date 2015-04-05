function res = fastExponentiationMod(base, pow, modulus)
% Возвращает (base ^ pow) mod (modulus)
    res = 1;
    while pow > 1
        if mod(pow, 2) == 1
            res = mod(res * base, modulus);
            pow = pow - 1;
        end
        base = mod(base * base, modulus);
        pow = pow / 2;
    end
end
