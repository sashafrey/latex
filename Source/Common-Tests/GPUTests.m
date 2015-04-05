function GPUTests
%   The test will fail if you don't have NVidia CUDA-compatible GPU devices (like, for example, GeForce GTX 660-ti ;)).
%   Description of test coverage:
%     calcAlgs and calcAlgsEV give the same result
%     createSession, getSessionStats, closeSession, closeAllSessions - can call without issues
%     startRecording, stopRecording, replayRecording - work fine together
%     getXR - not tested
%     findNeighbors, findRandomNeighbors - indeed gives a neighbour (rho = 1)
%     findAllNeighbors, runRandomWalking - draws maps of distances between classifiers
%     performCrossValidation - compare overfitting with MC-calculation on host (in MatLab)
%     findSources - check that findSources is consistent with sources found by runRandomWalking
%     calcAlgsConnectivity - simple sanity checks for mean connectivity (connectivity>1, connectivity<Features)
%     calcQEpsCombinatorialEV and calcQEpsCombinatorial - consistency
%     Test calcQEpsCombinatorialEV on monotonic set

    GPU_Deinitialize();
    task = LoadTask('wine');
    [task, ~] = SplitTask(task, 0.2102);

    recordingFile = 'GPUTests_recording.dat';
    if (exist(recordingFile, 'file'))
       delete(recordingFile);
    end
    
    GPU_Initialize;    
    GPU_StartRecording(recordingFile);
    
    nRays = task.nFeatures * 6;
    R = rand(nRays, task.nFeatures) - 0.5;
    w0 = rand(1, task.nFeatures) - 0.5;
    
    vecW0 = rand(task.nFeatures * 10, task.nFeatures) - 0.5;

    maxAlgs = 200;
    maxIters = 10000;
    nCVIters = 10000;
    maxErrors = task.nItems;
    seed = 1;
    
    % test GPU_SetLogLevel
    GPU_SetLogLevel(0); % SILENT
    GPU_SetLogLevel(4); % DEBUG
    
    sessionId = GPU_CreateSession(task.objects, task.target - 1, R);
    [nItems, nFeatures, nRays, ~] = GPU_GetSessionStats(sessionId);
    Check(nItems == task.nItems);
    Check(nFeatures == task.nFeatures);
    Check(nRays == nRays);
    
    %% Test that all neighbors differs on exactly one item
    w0neigh = GPU_FindNeighbors(sessionId, w0);
    [~, ev0, ~, ~] = GPU_CalcAlgs(sessionId, w0);
    [~, ev1, ~, ~] = GPU_CalcAlgs(sessionId, w0neigh);
    for i=1:size(w0neigh, 1)
        if (~isnan(w0neigh(i, :)))
            Check(sum(ev0 ~= ev1(:, i)) == 1)
        end
    end
    
    %% Test that all random neighbors differs on exactly one item
    vecW1 = GPU_FindRandomNeighbors(sessionId, vecW0, seed);
    [~, ev0, ~, ~] = GPU_CalcAlgs(sessionId, vecW0);
    [~, ev1, ~, ~] = GPU_CalcAlgs(sessionId, vecW1);
    for i=1:size(vecW1, 1)
        if (~isnan(vecW1(i, :)))
            Check(sum(ev0(:, i) ~= ev1(:, i)) == 1)
        end
    end
    
    %% Draw the map of distances between classifiers for FindAllNeighbors
    W_allNeighbours = GPU_FindAllNeighbors(sessionId, w0, maxAlgs, maxIters, maxErrors);
    Check(~any(isnan(W_allNeighbours(:, 1))));
    [~, evAll, ecAll, hashesAll] = GPU_CalcAlgs(sessionId, W_allNeighbours);
    Check(length(unique(hashesAll)) == length(hashesAll));
    Check(all(sum(evAll)' == ecAll));
    
    maxVal = 20;
    evdAll = calcDist(evAll, 0, maxVal);
    imagesc(evdAll);            
    colormap(flipud(gray));

    % Draw the map of distances between classifiers for RunRandomWalking
    % Expectation is that rw-map has higher values than map of all
    % neighbours, and also that white color gradually populates from white
    % around diagonal towards black (in the corners).
    [W_rw, isSource_rw] = GPU_RunRandomWalking(sessionId, w0, maxAlgs, maxIters, maxErrors, 1.0, 1);
    [~, ev_rw, ec_rw, hashes_rw] = GPU_CalcAlgs(sessionId, W_rw);
    evd_rw = calcDist(ev_rw, 0, maxVal);
    imagesc(evd_rw);            
    colormap(flipud(gray));
    
    %% Check FindSources and GPU_CalcAlgsConnectivity 
    isSource_rw2 = GPU_FindSources(ev_rw);
    Check(all(isSource_rw == isSource_rw2));
    Check(sum(isSource_rw) > 0.05 * length(isSource_rw)); % not too few sources
    Check(sum(isSource_rw) < 0.95 * length(isSource_rw)); % not too much sources
    [upperCon, lowerCon] = GPU_CalcAlgsConnectivity(hashes_rw, ec_rw, nItems);
    Check(mean(upperCon) > 1);
    Check(mean(upperCon) < task.nFeatures);
    Check(mean(lowerCon) > 1);
    Check(mean(lowerCon) < task.nFeatures);

    %% Test that GPU_CalcAlgs and GPU_CalcAlgsEV gives the same results
    [ec_rw2, hashes_rw2] = GPU_CalcAlgsEV(ev_rw, task.target - 1);
    Check(all(ec_rw2 == ec_rw));
    Check(all(hashes_rw2 == hashes_rw));
    
    %% Test PerformCrossValidation
    [~, ~, trainErrorRate, testErrorRate] = GPU_PerformCrossValidation(ev_rw, ec_rw, nCVIters);
    overfitting = testErrorRate - trainErrorRate;
    [QEps, ~, ~, ~, allOverfittings]  = CalcOverfitting(AlgsetCreate(ev_rw'==1), 0.5, nCVIters/10, 1, 0.1, 0.01);
    
    % plot overfittings
    hold on; 
    plot(sort(overfitting)); 
    plot(1:length(allOverfittings), sort(allOverfittings), 'r'); 
    hold off;
    
    % print QEps values
    %for i=1:length(QEps.X)
    %    fprintf('%.3f, %.3f\n', mean(overfitting >= QEps.X(i)), QEps.Y(i));
    %end
    
    %BUG-BUG (GPU version only). From both (1) and (2) it seems that calculation via GPU
    %slightly overestimates the overfitting (but just a little bit).
    
    %% Test consistency between GPU_CalcQEpsCombinatorial and its -EV version
    for boundType = [0, 1, 2]
        [QEps, eps] = GPU_CalcQEpsCombinatorial(sessionId, W_rw, isSource_rw, boundType);
        [QEps2, eps2] = GPU_CalcQEpsCombinatorialEV(ev_rw, ec_rw, hashes_rw, isSource_rw, boundType);
        
        Check(all(QEps(:) == QEps2(:)));
        Check(all(eps == eps2));    
        QEpsAll{boundType + 1} = QEps;
        
    end
    
    % Test that the format of clusterIds doesn't crash GPU_CalcQEpsCombinatorial
    clusterIds = GPU_DetectClusters(ev_rw, ec_rw);
    GPU_CalcQEpsCombinatorialAF(ev_rw, ec_rw, hashes_rw, isSource_rw, clusterIds);
    clusterIds = 0:(length(ec_rw) - 1);
    [QEps4, eps4] = GPU_CalcQEpsCombinatorialAF(ev_rw, ec_rw, hashes_rw, isSource_rw, clusterIds');
    QEpsAll{4} = QEps4;

    plot([sum(QEpsAll{1});sum(QEpsAll{2});sum(QEpsAll{3});sum(QEpsAll{4})]')
    
    %% Test testCalcQEpsCombinatorial(sessionId);
    testCalcQEpsCombinatorial();
    
    %% Simple test for getSources - test that for randomly-generated set all algs are sources.
    testSources();

    %% Test replayer
    GPU_CloseSession(sessionId);
    GPU_CloseAllSessions();
    
    GPU_StopRecording;
    GPU_ReplayRecording(recordingFile); % throw if fails.
    delete(recordingFile);
end

function evd = calcDist(ev, doSort, maxVal)
    nAlgs = size(ev, 2);
    evd = zeros(nAlgs, nAlgs);
    for i=1:nAlgs
        for j=i+1:nAlgs
            err = sum(ev(:, i) ~= ev(:, j));
            evd(i, j) = err;
        end
    end
    
    evd = evd + evd' + eye(nAlgs) * maxVal;
    
    if (doSort)
        [~, id] = sort(evd(1, :));
        evd2 = zeros(nAlgs, nAlgs);
        for i=1:nAlgs
            evd2(i, :) = evd(id(i), :);
        end
        evd = evd2;

        for i=1:nAlgs
            evd2(:, i) = evd(:, id(i));
        end
        evd = evd2;
    end
end

function testCalcQEpsCombinatorial()
    nItems = 100;
    algset = GenerateMonotonicSet(nItems, 5, 11, 1);
    ev = AlgsetGet(algset, 1:algset.Count)';
    [ec, hashes] = GPU_CalcAlgsEV(ev, zeros(nItems, 1));
    sources = GPU_FindSources(ev);
    Check(sum(sources) == 1); % only one sources in monotonic set
    [QEpsGPU_bt0, ~] = GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 0);
    [QEpsGPU_bt1, ~] = GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 1);
    [QEpsGPU_bt2, epsValues] = GPU_CalcQEpsCombinatorialEV(ev, ec, hashes, sources, 2);
    
    clusterIds = 0:(algset.Count - 1);
    [QEpsGPU_bt3, ~] = GPU_CalcQEpsCombinatorialAF(ev, ec, hashes, sources, clusterIds');
    %Check(max(max(abs(QEpsGPU_bt1 - QEpsGPU_bt3))) < 1e-5);        
    
    Check(all(abs(sum(QEpsGPU_bt0) - sum(QEpsGPU_bt1)) < 1e-5));
   % Check(all(abs(sum(QEpsGPU_bt3) - sum(QEpsGPU_bt1)) < 1e-5));
    
    edges = BuildHasseGraph(algset);
    [algset, edges] = BuildInternalClosure(algset, edges);
    ell = floor(algset.L / 2);
    k = algset.L - ell;
    QEpsH.X = []; QEpsH.Y = [];
    for eps = epsValues
        [Q, ~] = CalcQEpsHasse(algset, edges, eps, ell, k );
        QEpsH.X = [QEpsH.X; eps];
        QEpsH.Y = [QEpsH.Y; Q];    
    end

    hold on;
    plot(epsValues, sum(QEpsGPU_bt0));
    plot(QEpsH.X, QEpsH.Y);
    hold off;
end

function testSources() 
    nItems = 10;
    nAlgs = cnk(10, nItems / 2);
    algs = false(int32(nAlgs), nItems);
    algs1 = allsamples(nItems, nItems / 2);
    for i=1:nAlgs
        algs(i, algs1(i, :))= true;
    end  
    
    sources = GPU_FindSources(algs');
    Check(all(sources));
end