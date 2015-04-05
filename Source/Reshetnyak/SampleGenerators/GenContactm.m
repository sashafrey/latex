xst = [0 : 6 : 10];
n = size(xst, 2);
yst = repmat([2 8], 1, n /2 + 1);
d = 10 * rand(186, 2);

cl = ones(186, 1);
for i = 1:186
    for j = 1:n
        if (xst(j) > d(i, 1))
            if( yst(j) > d(i, 2) ) 
                cl(i) = 2;
            end
            break;
        end
    end
end

fp = zeros(186, 3);

fp(:, 1) = cl;
fp(:, 2) = d(:, 1)
fp(:, 3) =  d(:, 2)

save trainset.txt fp -ascii



nd = 10 * rand(500, 2);

save newdata.txt nd -ascii

