function res = CnkCalc(n, k)
    mlock();
    persistent binCoef_CnkCalc;
    
    if ((n + 1) > size(binCoef_CnkCalc, 1))
        binCoef_CnkCalc = CnkCreate(n);
    end
    
    if (n >= 0 && k >= 0 && k <= n)
        res = binCoef_CnkCalc(n + 1, k + 1);
    else
        res = 0;
    end
end

function binCoef = CnkCreate(L)
    binCoef = zeros(L + 1, L + 1);
    binCoef(:, 1) = 1;
    binCoef(1, 2:end) = 0;
    for i = 1:L
        for j = 1:L
            binCoef(i + 1, j + 1) = binCoef(i, j + 1) + binCoef(i, j);
        end
    end
end
