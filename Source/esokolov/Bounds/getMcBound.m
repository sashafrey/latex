function bound = getMcBound(algs, L, ell, nSplits)
% Вычисление оценки переобученности методом Монте-Карло по окрестности алгоритма w.

%     L = length(algs.Data(1).errVect);
    nAlgs = algs.Count;

    errMatrix = zeros(nAlgs, L);
    for i = 1:nAlgs
        errMatrix(i, :) = algs.Data(i).errVect;
    end
    
    % монте-карло
    bound = 0;
    
    for currSplit = 1:nSplits
        trainIdx = randsample(L, ell);
        testIdx = setdiff(1:L, trainIdx);
        
        trainErr = sum(errMatrix(:, trainIdx), 2);
        testErr = sum(errMatrix(:, testIdx), 2);
        
        bestTrainAlgsIdx = find(trainErr == min(trainErr));
        resAlg = bestTrainAlgsIdx(testErr(bestTrainAlgsIdx) == ...
            max(testErr(bestTrainAlgsIdx)));
        resAlg = resAlg(1);
        
%         bound = bound + ...
%             (testErr(resAlg) / (L - ell) - trainErr(resAlg) / ell);
        bound = bound + testErr(resAlg) / (L - ell);
    end
    
    bound = bound / nSplits;        
end
