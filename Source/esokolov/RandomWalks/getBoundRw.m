function bound = getBoundRw(X, Y, w, ell, eps, boundType)
% Вычисление оценки вероятности переобучения по окрестности алгоритма w.
% Требование: в выборке X есть константный признак, и ему соответствует
% последний столбец в X

    if ~exist('boundType', 'var')
        boundType = 'SC_sources';
    end
    
    L = size(X, 1);
    d = size(X, 2) - 1;

    % переводим линейный классификатор <w, x> в удобное для нас
    % представление
%     alg_start = convertLinearAlgToStructure(w, X, Y);
    alg_start = convertLinearAlgToSimpleStructure(w, X, Y);
    
    % запускаем случайное блуждание из alg_start
    iterCnt_rw = 500;
    maxLevel_rw = alg_start.errCnt + 15;
%     [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
%         alg_start, iterCnt_rw, maxLevel_rw, @getNeighboursLC_rw, true);
    [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
        alg_start, iterCnt_rw, maxLevel_rw, ...
        @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, 50), ...
        true);
    
    % для дальнейших вычислений нам понадобится оценка на V_t --- число
    % вершин в слоях с 1-го по maxLevel_rw
    iterCnt_Vt = 10000;
    maxLevel_Vt = maxLevel_rw;
    %Vt = getVt(X, Y, iterCnt_Vt, maxLevel_Vt, false);    
    % ВРЕМЕННО!!!
    Vt = 1;
    
    % истинных истоков мы скорее всего не нашли, поэтому примем за истоки
    % наилучшие алгоритмы из algs_rw
    sources = findSourcesInSample(algs_rw);
    
    % вычисляем оценку по выборке алгоритмов, полученной в результате
    % случайного блуждания
    bound = getSrwBoundEstimate(algs_rw, corrections_rw, sources, L, ell, ...
        eps, Vt, boundType);
end
