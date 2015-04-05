 function [res] = HypergeometricSum(L, ell, m, lb, rb, chooseTable)
    res = 0;
    if (ell < 0 || L < ell || lb < 0 || rb > ell)
        return
    end

    if nargin < 6
        for i = max(lb, m + ell - L) : min(rb, m)
            res = res + nchoosek(m, i) * nchoosek(L - m, ell - i + 1);
        end
    else
        for i = max(lb, m + ell - L) : min(rb, m)
            res = res + chooseTable(m + 1, i + 1) * chooseTable(L - m + 1, ell - i + 1);
        end
    end
end