c1 = 7*rand(2, 20);
c2 = 7*rand(2, 20) + 8;
fp = zeros(3, 40);
fp(1, :) = [repmat(1, 1, 20) repmat(2, 1, 20)];
fp(2, :) = [c1(1, :) c2(1, :)];
fp(3, :) = [c1(2, :) c2(2, :)];
fp = fp'

save  sample.txt fp -ascii


c3 = 12 * rand(50, 2)
save newdata.txt c3 -ascii

size(fp)
size(c3)
