%function adjTables = CreateAdjustmentsTableAllTasks(nRules, adjTablesName, names, tasks)
    if (~exist('tasks', 'var'))
        tasks = LoadTasks;
    end
    
    params = [];
    params.nItersCV = 10000;
    params.parallel = true;
    params.verbose = true;
    params.max_features = 3;
    names = {'Echo_cardiogram', 'Australian', 'Heart_Disease', 'Hepatitis', 'Labor_Relations', 'Liver_Disorders', 'German'};

    %params.testMode = true; 
        %WARNING: With testMode=true you can't use produced
        %adjTables in COMBoost_Evaluation, because the will be only built for 10
        %random sets of features. COMBoost_Evaluation requires all adjTables for
        %all combinations of up to params.max_features.
    %params.nItersCV = 10000;
    %params.T1 = 5;
    %params.parallel = false;
    params.nTopRules = 250;
    names = {'Hepatitis', 'Labor_Relations', 'Heart_Disease', 'Echo_cardiogram' };

    adjTables = [];
    for name_cell = names
        taskname = name_cell{1};
        fprintf('%s...\n', taskname)
        adjTables_taskname = CreateAdjustmentsTable(tasks.(taskname), params);
        adjTables.(taskname) = adjTables_taskname;
        save('adjTables', 'adjTables')
        fprintf('%s done.\n', taskname)
    end
%end