function [] = OneLevelEstimation( )

function [res] = G(s, L, l, m)
        res = 0;
        if (l < 0 || L < l)
            return
        end
        for i = max(0, m + l - L) : min(s, m)
            res = res + cTable(m + 1, i + 1) * cTable(L - m + 1, l - i + 1);
        end
 end

L = 500;
l = L/2;
k = L - l;
eps = 0.05;

cTable = zeros(L + 1, L + 1);
for i = 1:L+1
    cTable(i, 1) = 1;
     for j = 2:i 
         cTable(i, j) = cTable(i - 1, j - 1) + cTable(i - 1, j);
     end
end

gRes = zeros(1, L);
gInt = zeros(1, L);
svals = floor(l/L * ([1:L] - k * eps) );
ind = find(svals >= 0);
for m = ind
    gRes(m) = G(svals(m), L, l, m);
    gInt(m) = G(svals(m), L - 2, l, m - 1) + 2 * G(svals(m) - 1, L - 2, l - 1, m - 1) + G(svals(m) - 1, L - 2, l - 2, m - 1);
end

gRes = gRes / nchoosek(L, l);
gInt = gInt / nchoosek(L, l);

hold on
% plot([1:L], gRes, 'g');
% plot([1:L], gInt, 'r');
% plot([1:L], gRes - gInt, 'b');
plot([1:L], 1 - gInt./ gRes);

end

