function p = AddNoiseFeature(p, n)
    r = p.rand;
    sample = r.rand(p.L,n);
    p.sample = [p.sample(:,1:end-1), sample, p.sample(:,end)];
    p.n = p.n+n;
end