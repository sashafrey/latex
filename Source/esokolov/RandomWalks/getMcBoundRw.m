function bound = getMcBoundRw(X, Y, w, ell, nSplits)
% Вычисление оценки переобученности методом Монте-Карло по окрестности алгоритма w.

    L = size(X, 1);
    d = size(X, 2);

    % переводим линейный классификатор <w, x> в удобное для нас
    % представление
%     alg_start = convertLinearAlgToStructure(w, X, Y);
    alg_start = convertLinearAlgToSimpleStructure(w, X, Y);
    
    % запускаем случайное блуждание из alg_start
    iterCnt_rw = 1000;
    %maxLevel_rw = alg_start.errCnt + 15;
    maxLevel_rw = alg_start.errCnt + 150;
%     [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
%         alg_start, iterCnt_rw, maxLevel_rw, @getNeighboursLC_rw, true);
    [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
        alg_start, iterCnt_rw, maxLevel_rw, ...
        @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, 50), ...
        true);
    
    errMatrix = zeros(iterCnt_rw, L);
    for i = 1:iterCnt_rw
        errMatrix(i, :) = algs_rw.Data(i).errVect;
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
        
        bound = bound + ...
            (testErr(resAlg) / (L - ell) - trainErr(resAlg) / ell);
    end
    
    bound = bound / nSplits;        
end
