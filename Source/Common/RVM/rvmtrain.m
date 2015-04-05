function model = rvmtrain(task, params)
    Check(task.nClasses == 2);
    
    if (~exist('params', 'var'))
        params = [];
    end
    
    % Set verbosity of output (0 to 4)
    setEnvironment('Diagnostic','verbosity',0);
    % Set file ID to write to (1 = stdout)
    setEnvironment('Diagnostic','fid',1);

    params = SetDefault(params, 'width', 0.5);
    params = SetDefault(params, 'kernel', 'gauss');
    params = SetDefault(params, 'maxItems', 500);
    params = SetDefault(params, 'maxIts', 500);
    params = SetDefault(params, 'monIts', 100);
    params = SetDefault(params, 'useBias', true);    

    if (task.nItems > params.maxItems)
        task = GetTaskSubsample(task, randsampleStratified(task.target, params.maxItems));
    end

    N           = task.nItems;
    initAlpha	= (1/N)^2;
    initBeta	= 0;

    X = task.objects;
    X(isnan(X)) = 0; % fill gaps with 0s.
    t = task.target - 1; 
        
    [weights, used, bias, ml, alpha, beta, gamma] = ...
        SB1_RVM(X, t, initAlpha, initBeta, params.kernel, params.width, params.useBias, params.maxIts, params.monIts);
    
    model.weights = weights;
    model.used = used;
    model.bias = bias;
    model.ml = ml;
    model.alpha = alpha;
    model.beta = beta;
    model.gamma = gamma;
    
    model.X = X; % save train dataset.
    model.kernel = params.kernel;
    model.width = params.width;
    model.useBias = params.useBias;
end