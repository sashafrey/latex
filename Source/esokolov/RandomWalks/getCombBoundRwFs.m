function bound = getCombBoundRwFs(X, Y, w_arr, ell, iterCnt_rw, maxLevel_rw)
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
%     [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
%         alg_start, iterCnt_rw, maxLevel_rw, @getNeighboursLC_rw, true);
    [algs_rw, corrections_rw] = random_walk_fs(X, Y, ...
        alg_start_arr, iterCnt_rw, maxLevel_rw, ...
        @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, 100), ...
        true);
    
   
    % истинных истоков мы скорее всего не нашли, поэтому примем за истоки
    % наилучшие алгоритмы из algs_rw
    sources = findSourcesInSample(algs_rw);
    
    % вычисляем оценку по выборке алгоритмов, полученной в результате
    % случайного блуждания
%     boundType = 'SC_sources';
    boundType = 'CCV_classic';
%     bound = getSrwBoundEstimate(algs_rw, corrections_rw, sources, L, ell, ...
%         eps, Vt, boundType);      
    bound = getLayeredBoundEstimate(algs_rw, corrections_rw, sources, L, ell, ...
        0.1, boundType);
end
