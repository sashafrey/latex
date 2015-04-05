function GPUTestsRandomWalkingPTransition()
    task = LoadTask('wine');
    [task, ~] = SplitTask(task, 0.05);
    nRays = task.nFeatures * 6;
    nAlgs = 1;
    maxAlgs = 100;
    maxIters = 1000;
    R = rand(nRays, task.nFeatures) - 0.5;
    w0 = ones(nAlgs, 1) * (rand(1, task.nFeatures) - 0.5);
    
    sessionId = GPU_CreateSession(task.objects, task.target - 1, R);
    maxErrors = task.nItems;    
    
    %% Draw error counts of RunRandomWalking at different level of pTransition
    pTransitions = [0.1, 0.2, 0.5, 0.75, 0.9];
    ecAll = NaN(length(pTransitions), maxAlgs);
    for i=1:length(pTransitions) 
        [W_rw, isSource_rw] = GPU_RunRandomWalking(sessionId, w0, maxAlgs, maxIters, maxErrors, pTransitions(i), 0);
        [~, ev_rw, ec_rw, hashes_rw] = GPU_CalcAlgs(sessionId, W_rw);
        ecAll(i, 1:length(ec_rw)) = ec_rw;
    end
    plot(ecAll')
    
    plot(Aggregate(ecAll, 10)')
end

function ecAllAgg = Aggregate(ecAll, window)
	n = size(ecAll, 2);
    nAgg = n / window;
    ecAllAgg = zeros(size(ecAll, 1), nAgg);
    for i=1:nAgg
        start = min(1 + (i-1)*window, n);
    	finish = min(1 + i * window, n);
        ecAllAgg(:, i) = mean(ecAll(:, start:finish), 2);
    end
end
