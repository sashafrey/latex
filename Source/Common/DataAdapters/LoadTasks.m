function tasks = LoadTasks(folder)
    if (nargin == 0)
        folder = LocateDataUCI();
    end
    
    tasks = struct();
    folders = dir(folder);
    for taskDir = folders'
        if (~taskDir.isdir) continue; end;
        if (taskDir.name == '.') continue; end;
        fprintf('%s\n', taskDir.name)
        try
            task = LoadTask(fullfile(folder, taskDir.name));
        catch e
            fprintf('Unable to load task %s. Exception %s', taskDir.name, e.message);
            continue;
        end
        
        if (task.nClasses > 2)
            continue;
        end
        
        tasks.(taskDir.name) = task;
    end
end