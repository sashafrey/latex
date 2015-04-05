function [res] = HypergeometricTail(L, ell, m, s, chooseTable)
    if nargin < 5
        res = HypergeometricSum(L, ell, m, 0, s);
    else
        res = HypergeometricSum(L, ell, m, 0, s, chooseTable);
    end
end