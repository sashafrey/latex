function bound = getMcBoundRwFs(X, Y, w_arr, ell, ...
    iterCnt_rw, maxLevel_rw, nSplits)
% Вычисление оценки переобученности методом Монте-Карло по окрестности алгоритма w.

    L = size(X, 1);
    d = size(X, 2);

    % переводим линейный классификатор <w, x> в удобное для нас
    % представление
%     alg_start = convertLinearAlgToStructure(w, X, Y);
    for i = 1:length(w_arr)
        alg_start_arr(i) = convertLinearAlgToSimpleStructure(w_arr{i}, X, Y);
    end
    
    % запускаем случайное блуждание из alg_start
    
%     maxLevel_rw = 0;
%     for i = 1:length(w_arr)
%         maxLevel_rw = max(maxLevel_rw, alg_start_arr(i).errCnt);
%     end
%     maxLevel_rw = maxLevel_rw + 10;
    
    
    
    %maxLevel_rw = alg_start_arr(1).errCnt + 15;
    %maxLevel_rw = alg_start_arr(1).errCnt + 150;
%     [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
%         alg_start, iterCnt_rw, maxLevel_rw, @getNeighboursLC_rw, true);
    [algs_rw, corrections_rw] = random_walk_fs(X, Y, ...
        alg_start_arr, iterCnt_rw, maxLevel_rw, ...
        @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, 50), ...
        true);
    
%     % типа убираем алгоритмы, отсекаемые регуляризацией
%     eta = sum(w_train .^ 2);
%     for i = algs_rw.Count:-1:1
%         currNorm = sum(algs_rw.Data(i).w .^ 2);
%         if currNorm > eta
%             %algs_rw(i) = [];
%             algs_rw = StructVectorDel(algs_rw, i);
%         end
%     end
    
    errMatrix = zeros(iterCnt_rw, L);
    for i = 1:algs_rw.Count
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
        
%         bound = bound + ...
%             (testErr(resAlg) / (L - ell) - trainErr(resAlg) / ell);
        bound = bound + testErr(resAlg) / (L - ell);
    end
    
    bound = bound / nSplits;        
end
