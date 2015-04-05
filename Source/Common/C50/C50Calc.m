function result = C50Calc(composition, task)
    ConvertTaskToC50DataFile(task, task.filename, '.cases', 'C:\\temp');    
    
    command = sprintf('C:\\temp\\C50Calc.exe -f C:\\temp\\%s -s', task.filename);
    
    if (composition.rules)
        command = sprintf('%s -r', command);
    end
    
    [status, consoleOutput] = system(command);    
    if (status ~= 0)
        fprintf('ERROR: %s failed.\n', command);
        fprintf('%s\n', consoleOutput);
        result = [];
        return;
    end

    foutputfile = sprintf('C:\\temp\\%s.out', task.filename);
    result = dlmread(foutputfile);
    
    if (isempty(result))
        fprintf('Unable to parse result from %s. Console output: %s\n', foutputfile, consoleOutput);
    end
end
