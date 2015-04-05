function [overfitProb, chooseCoeffs] = ComputeOverfitProbViaInclusionExclusion(algs, trainSize, errorLevel)

    %Filter algorithms
    algs = algs( sum(algs, 2) >= errorLevel, :);
    [numAlgs sampleSize] = size(algs);
    chooseTable = ComputeChooseTable(sampleSize);
    chooseCoeffs = zeros(1, sampleSize + 1);
    
    overfitProb = 0;
    factor = -1;
    for n = 1:numAlgs
        ind = nchoosek(1:numAlgs, n);
        factor = -factor;
        for i = 1:size(ind, 1)
            if n == 1
                t = sampleSize + 1 - sum( algs(ind(i, :), :)  > 0);
            else
                t = sampleSize + 1 - sum( sum(algs(ind(i, :), :) ) > 0);
            end
            overfitProb = overfitProb + factor * chooseTable(t , trainSize + 1) ;
            chooseCoeffs(t) = chooseCoeffs(t) + factor;
        end
        
    end
    
    overfitProb = overfitProb / chooseTable(sampleSize + 1, trainSize + 1);
    
end
