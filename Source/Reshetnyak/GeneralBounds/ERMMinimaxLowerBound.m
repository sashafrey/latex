function [res] = ERMMinimaxLowerBound(sampleSize, trainSize, eps, dim, cTable)
    if nargin < 5
        cTable = ComputeChooseTable(sampleSize);
    end
    
    res = 0;
    testSize = sampleSize - trainSize;
    for s = 0 : trainSize - dim
        curSum = 0;
        for r = ceil( testSize * s / trainSize + testSize * eps) : testSize
            curSum = curSum + cTable(testSize + 1, r + 1);
        end
        if s == 0
            res = res + curSum * sum(cTable(trainSize + 1, 1 : dim + 1));
        else
            res = res + curSum * cTable(trainSize + 1, dim + s + 1);
        end
    end
        
    res = res / 2^sampleSize
end