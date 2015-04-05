
x =[];
y =[];
classes = [];
for nc = 1:4
    for r = 3*(nc - 1) : 0.3 : 3*nc
        ang = [0: pi/6 :2*pi];
        x = [x r*cos(ang)] ;
        y = [y r*sin(ang)] ;
        classes = [classes repmat(nc, 1,  size(ang, 2) )]
    end
end
 
n = size(classes, 2);
fp = zeros(n, 3)
fp(:, 1) = classes
fp(:, 2) = x + 0.3 * rand(1, n);
fp(:, 3) = y + 0.3 * rand(1, n);

save sample.txt fp -ascii



nd = 24 * (rand(700, 2) - 0.5)
size(fp)
size(nd)
save newdata.txt nd -ascii