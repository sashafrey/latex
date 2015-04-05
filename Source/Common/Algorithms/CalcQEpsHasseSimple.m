function [Q, P, C] = CalcQEpsHasseSimple(algset, edges, eps, ell, k )
    % without upper connectivity.
    L = algset.L;
    assert((ell + k) == L);
    
    nAlgs = algset.Count;
        
    CLl = CnkCalc(L, ell);
    
    Q = 0;
    P = 0;
    C = 0;
    for i = 1:nAlgs
        alg = AlgsetGet(algset, i);
        
        
        parentsIntersection = true(1, L);
        allParents = GetAllParents(edges, i);
        for j=1:length(allParents)
            parent = AlgsetGet(algset, allParents(j));
            parentsIntersection = (parentsIntersection & parent);
        end
        assert(all(parentsIntersection <= alg));
        
        m = sum(alg);
        qa = m - sum(parentsIntersection); 
        La = L - qa;
        ma = m - qa;
        sa = ell / L * (m - eps * k);
        
        curP = CnkCalc(La, ell) / CLl;
        curH = hhDistr(La, ell, ma, sa);
        curC = m - ell / La * ma;

        Q = Q + curP * curH;
        P = P + curP;
        C = C + curP * curC;
    end
end