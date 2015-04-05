function bound = calcScBoundFromScProfile(L, ell, eps, profile)
    binCoef = zeros(L + 1, L + 1);
    for i = 0:L
        for j = 0:L
            binCoef(i + 1, j + 1) = cnk(i, j);
        end
    end

    bound = 0;
    contributions = profile;
    for m = 0:L
        for q = 0:L
            for r = 0:L
                if (profile(m + 1, q + 1, r + 1) == 0)
                    continue;
                end
                contributions(m + 1, q + 1, r + 1) = profile(m + 1, q + 1, r + 1) * (my_cnk(L - q - r, ell - q, binCoef) / my_cnk(L, ell, binCoef)) * ...
                    my_hhDistr(L - q - r, ell - q, m - r, floor((ell/L) * (m - eps * (L - ell))), binCoef);
                bound = bound + contributions(m + 1, q + 1, r + 1);
            end
        end
    end
end

function res = my_cnk(n, k, binCoef)
    if (n >= 0 && k >= 0 && k <= n)
        res = binCoef(n + 1, k + 1);
    else
        res = 0;
    end
end

function h = my_hhDistr(L, l, m, s, binCoef)
    h = 0;
    if (L >= 0 && l >= 0 && l <= L)
        for i = 0:s
            h = h + my_cnk(m, i, binCoef) * my_cnk(L - m, l - i, binCoef) / my_cnk(L, l, binCoef);
        end
    end
end