L = 100;
ell = 50;
k = L - ell;
m = 10;
ind = 1;
for d = 0:1:10
    algset = false(2, L);    
    algset(:, 1:m) = true;
    algset(1, (m+1)   : (m+d)) = true;
    algset(2, (m+d+1) : (m+2*d)) = true;
    [~, eps, ~, ~] = CalcOverfitting(AlgsetCreate(algset), 0.5, 50000, 0.05, 0.5, 0.01);
    q.d(ind) = d;
    q.eps(ind) = eps;
    ind = ind + 1;
end

