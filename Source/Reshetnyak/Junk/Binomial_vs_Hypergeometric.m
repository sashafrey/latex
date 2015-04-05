L = 400;
l = 200;
m = 20;
delta = 0.1;
% S = repmat([0:l], l + 1, 1);
s = [1:20];
l = 2 * s - 1;

H = hygecdf(zeros( size(s)), 2*l, s, l)
% H = hygecdf( S, L, S' , l);
% ht = zeros(1, m + 1);
% for i = 0:m
%     %find(H(:,i + 1) < delta, 1) + 1
%     ht(i + 1) = (find(H(:,i + 1) < delta, 1) - 1 - i) / (L - l);
% end
% hold on
% plot([0:m], ht,'g');
% find(H < delta);
% B = binocdf(S,l,repmat([0:1/l:1], l + 1, 1)');
% 
% bt = zeros(1, m + 1);
% for i = 0:m
%     %find(B(:,i + 1) < delta, 1) + 1
%     bt(i + 1) = (find(B(:,i + 1) < delta, 1) - 1) * 1/l;
% end
% plot([0:m], bt, 'r');
% 
% delta = 0.05
% r = binornd(l, 0.1)
% [phat, pci] = binofit(r, l, delta/2)
% pci
% rp = binoinv(1 - delta/4, l, pci(2) ) / l
% lp = binoinv(delta/4, l, pci(1)) / l
% 
% %[temp h1] = min( find(H )
% %plot(, H);