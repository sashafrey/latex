function algset = GenerateMonotonicSet(L, D, m, step)
    H = 2;
    nAlgs = (D+1)*H;
    nItems = D * H * step + m;
    assert(L >= nItems);
    algset = false(nAlgs, L);
    ind = 1;
    for i1=0:D
        for i2=0:D        
            algset(ind, (0*D*step + 1):(0*D*step + i1*step)) = true;
            algset(ind, (1*D*step + 1):(1*D*step + i2*step)) = true;
            ind = ind + 1;
        end
    end
    
    algset(:, (nItems - m + 1) : nItems) = true;
    algset = AlgsetAdd(AlgsetCreate(), algset);
end
