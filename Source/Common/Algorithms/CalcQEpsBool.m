function [Q, P, C] = CalcQEpsBool(algset, edges, eps, ell, k )
    L = algset.L;
    assert((ell + k) == L);
    nAlgs = algset.Count;
    Q = 0;
    C = 0;
    for i = 1:nAlgs
        m = sum(AlgsetGet(algset, i));
        s = ell / L * (m - eps * k);
        Q = Q + hhDistr(L, ell, m, s);
        C = C + m;
    end
    
    P = nAlgs;
end