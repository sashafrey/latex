function [Q, P, C] = CalcQEpsHasse(algset, edges, eps, ell, k )
    L = algset.L;
    assert((ell + k) == L);
    
    nAlgs = algset.Count;
    childrenCount = zeros(1, nAlgs);
    for i = 1:nAlgs
        childrenCount(i) = size(GetChildren(edges, i), 2);
    end
    maxChildrenCount = max(childrenCount);
    
    sau_storage = CreateSAU(maxChildrenCount);
    CLl = CnkCalc(L, ell);
    
    Q = 0;
    P = 0;
    C = 0;
    for i = 1:nAlgs
        alg = AlgsetGet(algset, i);
        children = GetChildren(edges, i);
        
        nChildren = size(children, 2);
        v = zeros(1, maxChildrenCount);
        for j=1:nChildren
            child = AlgsetGet(algset, children(j));
            v(j) = sum((child == 1) & (alg == 0));
        end
           
        parentsIntersection = true(1, L);
        allParents = GetAllParents(edges, i);
        for j=1:length(allParents)
            parent = AlgsetGet(algset, allParents(j));
            parentsIntersection = (parentsIntersection & parent);
        end
        assert(all(parentsIntersection <= alg));
        
        m = sum(alg);
        qa = m - sum(parentsIntersection); 
        ua = nChildren;
        La = L - qa;
        ell_a = ell - ua;
        ma = m - qa;
        sa = ell / L * (m - eps * k);
        
        for u = ua:sum(v)
            [sau, sau_storage] = CalcSAU([u - ua, v - 1], sau_storage);
            curP = CnkCalc(La - u, ell_a) / CLl;
            curH = hhDistr(La - u, ell_a, ma, sa);
            curC = m - ell_a / (La - u) * ma;
            
            Q = Q + sau * curP * curH;
            P = P + sau * curP;
            C = C + sau * curP * curC;
        end
    end
end