function composition = C50Tune(task, params)
    %  Summary of options for c5.0:
    %  -f <filestem>   application filestem
    %  -r              use rule-based classifiers
    %  -u <bands>      order rules by utility in bands
    %  -w              invoke attribute winnowing
    %  -b              invoke boosting
    %  -t <trials>     number of boosting trials
    %  -p              use soft thresholds
    %  -e              focus on errors (ignore costs file)
    %  -s              find subset tests for discrete atts
    %  -g              do not use global tree pruning
    %  -m <cases>      restrict allowable splits
    %  -c <percent>    confidence level (CF) for pruning
    %  -S <percent>    training sample percentage
    %  -X <folds>      cross-validate
    %  -I <integer>    random seed for sampling and cross-validation

    if (~exist('params', 'var'))
        params = [];
    end

    C50Init();
        
    params = SetDefault(params, 'rules', false); % default = trees.
    params = SetDefault(params, 'winnowing', false);
    params = SetDefault(params, 'boost', false);
    params = SetDefault(params, 'trials', 0);
    params = SetDefault(params, 'softThresholds', false);
    params = SetDefault(params, 'pruningCF', 0);          % use C50 default.
    params = SetDefault(params, 'trainSamplePercent', 0); % use C50 default.

    ConvertTaskToC50DataFile(task, task.filename, '.data', 'C:\\temp');

    command = sprintf('C:\\temp\\C50Tune.exe -f C:\\temp\\%s', task.filename);
    
    if (params.rules)
        command = sprintf('%s -r', command);
    end
    
    if (params.winnowing)
        command = sprintf('%s -w', command);
    end
    
    if (params.boost)
        command = sprintf('%s -b', command);
    end
    
    if (params.trials > 0)
        command = sprintf('%s -t %d', command, params.trials);
    end
    
    if (params.softThresholds)
        command = sprintf('%s -p', command);
    end
    
    if (params.pruningCF > 0)
        command = sprintf('%s -c %d', command, params.pruningCF);
    end
    
    if (params.trainSamplePercent > 0)
        command = sprintf('%s -S %d', command, params.trainSamplePercent);
    end
    
    [status, result] = system(command);
    
    if (status ~= 0)
        fprintf('%s failed.\n', command);
        fprintf('%s\n', result);
        composition = [];
        return;
    end
    
    composition = params;
end