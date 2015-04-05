%%
clear;

%%
load('./esokolov/Experiments/RW/unsep200_rw_ready.mat');

%% все наши случайные блуждания
random_walkers = struct('func', cell(1), 'walker_name', cell(1));

random_walkers.func{1} = @(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display) ...
    random_walk_simple(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display, false);

random_walkers.func{2} = @(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display) ...
    random_walk_simple(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display, true);

random_walkers.func{3} = @(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display) ...
    random_walk_fs(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display, false);

random_walkers.func{4} = @(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display) ...
    random_walk_fs(X, Y, initialVertices, iterCnt, maxLevel, ...
    getNeighboursFuncHandle, display, true);

random_walkers.walker_name{1} = 'SRW';
random_walkers.walker_name{2} = 'SRW lazy';
random_walkers.walker_name{3} = 'FS';
random_walkers.walker_name{4} = 'FS lazy';

%% исправляем проблемы совместимости
sources_rw_old = algs_src(sources_src(1:sourcesCnt_src));
sources_rw_arr = VectorCreate();
for i = 1:length(sources_rw_old)
    sources_rw(i) = initLinearAlgStructure(sources_rw_old(i).vertex, ...
        sources_rw_old(i).errVect', sources_rw_old(i).errCnt, ...
        sources_rw_old(i).descr, ...
        sources_rw_old(i).lowerNeighsCnt, sources_rw_old(i).upperNeighsCnt, ...
        sources_rw_old(i).lowerNeighbours, sources_rw_old(i).upperNeighbours);
    sources_rw_arr = VectorAdd(sources_rw_arr, sources_src(i));
end

X = [X, ones(size(X, 1), 1)];
L = size(X, 1);

%% делаем случайные блуждания всеми методами
maxLevel_rw = maxLevel;

iterCnt_rw = 10000;
display_rw = true;
getNeightboursFunc = @getNeighboursLC_rw;

algs_rw_all = cell(length(random_walkers.func), 1);
corrections_rw_all = cell(length(random_walkers.func), 1);
parfor i = 1:length(random_walkers.func)
    [algs_rw_all{i}, corrections_rw_all{i}] = random_walkers.func{i}(X, Y, ...
        sources_rw, iterCnt_rw, maxLevel_rw, getNeightboursFunc, display_rw);
end

%% сохраняем результаты случайных блужданий в файл
save('./esokolov/Experiments/RW/unsep200_rw_res.mat', 'random_walkers', ...
    'algs_rw_all', 'maxLevel_rw', 'sources_rw', 'iterCnt_rw');

%% готовимся
ell_rw = 100;
eps_arr_rw = 0.01:0.01:0.3;
Vt_rw = Vt;
boundType_rw = 'SC_sources';
profileErr_rw = profileErr_true;
eps_rw = 0.1;
lengths_rw = 100:100:algs_rw_all{1}.Count;

%% вычисляем простые оценки для разных eps
bounds_simple_all_eps = zeros(length(random_walkers.func), length(eps_arr_rw));

for walker_idx = 1:length(random_walkers.func)
    parfor eps_idx = 1:length(eps_arr_rw)
        fprintf('%d %d\n', walker_idx, eps_idx);
        
        eps_curr = eps_arr_rw(eps_idx);
        
        bounds_simple_all_eps(walker_idx, eps_idx) = ...
            getSrwBoundEstimate(algs_rw_all{walker_idx}, ...
            corrections_rw_all{walker_idx}, sources_rw_arr, L, ell_rw, ...
            eps_curr, Vt_rw, boundType_rw);
    end
end

%% вычисляем послойные оценки для разных eps
bounds_layered_all_eps = zeros(length(random_walkers.func), length(eps_arr_rw));

for walker_idx = 1:length(random_walkers.func)
    parfor eps_idx = 1:length(eps_arr_rw)
        fprintf('%d %d\n', walker_idx, eps_idx);
        
        eps_curr = eps_arr_rw(eps_idx);
    
        bounds_layered_all_eps(walker_idx, eps_idx) = ...
            getLayeredBoundEstimate(algs_rw_all{walker_idx}, ...
            corrections_rw_all{walker_idx}, sources_rw_arr, ...
            L, ell_rw, eps_curr, boundType_rw);    
    end
end

%% вычисляем простые оценки для разных длин выборки
bounds_simple_all_len = zeros(length(random_walkers.func), length(lengths_rw));

for walker_idx = 1:length(random_walkers.func)
    parfor length_idx = 1:length(lengths_rw)
        fprintf('%d %d\n', walker_idx, length_idx);
        
        length_curr = lengths_rw(length_idx);
        
        bounds_simple_all_len(walker_idx, length_idx) = ...
            getSrwBoundEstimate(algs_rw_all{walker_idx}(1:length_curr), ...
            corrections_rw_all{walker_idx}(1:length_curr), ...
            sources_rw_arr, L, ell_rw, eps_rw, Vt_rw, boundType_rw);  
    end
end

%% вычисляем послойные оценки для разных длин выборки
bounds_layered_all_len = zeros(length(random_walkers.func), length(lengths_rw));

for walker_idx = 1:length(random_walkers.func)
    parfor length_idx = 1:length(lengths_rw)
        fprintf('%d %d\n', walker_idx, length_idx);
        
        length_curr = lengths_rw(length_idx);
    
        bounds_layered_all_len(walker_idx, length_idx) = ...
            getLayeredBoundEstimate(algs_rw_all{walker_idx}(1:length_curr), ...
            corrections_rw_all{walker_idx}(1:length_curr), ...
            sources_rw_arr, L, ell_rw, eps_rw, boundType_rw);    
    end
end

%% сохраняем еще и оценки
save('./esokolov/Experiments/RW/unsep200_rw_res.mat', 'random_walkers', ...
    'algs_rw_all', 'maxLevel_rw', 'sources_rw', 'iterCnt_rw', ...
    'bounds_simple_all_eps', 'bounds_layered_all_eps', ...
    'bounds_simple_all_len', 'bounds_layered_all_len', ...
    'eps_arr_rw', 'eps_rw');

%% нам понадобятся истинные оценки
% замечание: оценки на самом деле не очень истинные, так как мы при их
% вычислении используем не все истоки; однако, именно эти оценки мы
% приближаем в случайных блужданиях
algs_for_true_bound = algs_all(1:algsCnt);
for i = length(algs_for_true_bound):-1:1
    if algs_for_true_bound(i).errCnt > maxLevel
        algs_for_true_bound(i) = [];
    end
end

algs_for_true_bound_arr = StructVectorCreate(initLinearAlgStructure());
for i = 1:length(algs_for_true_bound)
    currAlg = initLinearAlgStructure(algs_for_true_bound(i).vertex, ...
        algs_for_true_bound(i).errVect', algs_for_true_bound(i).errCnt, ...
        algs_for_true_bound(i).descr, ...
        algs_for_true_bound(i).lowerNeighsCnt, algs_for_true_bound(i).upperNeighsCnt, ...
        algs_for_true_bound(i).lowerNeighbours, algs_for_true_bound(i).upperNeighbours);
    
    algs_for_true_bound_arr = StructVectorAdd(algs_for_true_bound_arr, ...
        currAlg);
end

bounds_true = zeros(length(eps_arr_rw), 1);
for eps_idx = 1:length(eps_arr_rw)
    eps_curr = eps_arr_rw(eps_idx);
    
    bounds_true(eps_idx) = getCombBound_ManyAlgs(algs_for_true_bound_arr, ...
        sourcesVects, L, ell_rw, eps_curr, boundType_rw);
end

save('./esokolov/Experiments/RW/unsep200_rw_res.mat', 'bounds_true', '-append');
