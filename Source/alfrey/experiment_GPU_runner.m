function experiment_GPU_runner
    tasks = LoadTasks();
    %tasknames = {'Magic04', 'Optdigits', 'glass', 'pageblocks', 'pendigits', 'pima', 'Ionosphere', 'Liver_Disorders', 'Sonar', 'Wdbc', 'faults', 'statlog', 'waveform', 'wine'};
    tasknames = {'Optdigits', 'glass', 'pageblocks', 'pima', 'Ionosphere', 'Liver_Disorders', 'Sonar', 'Wdbc', 'faults', 'statlog', 'waveform', 'wine'};
    for i=1:100
    for taskname = tasknames
            experiment_GPU(tasks.(taskname{1}));
    end
    end
end

