function ConvertTaskToC50DataFile( task, taskName, extension, outputFolder )
    % this converts task to the format acceptable by SEE (implementation of    
    % Quinlan's C5.0; http://www.rulequest.com/see5-win.html
    
    if (~exist('outputFolder', 'var'))
        outputFolder = '.';
    end
    
    if (~exist('extension', 'var'))
        extension = '.data';
        % alternative would be '.cases'.
    end
    
    if (~exist(outputFolder,'dir'))
        mkdir(outputFolder);
    end
    
    dataFile = fopen(fullfile(outputFolder, sprintf('%s%s', taskName, extension)), 'w');
    for i=1:task.nItems
        for j=1:task.nFeatures
            if (isnan(task.objects(i, j)))
                fprintf(dataFile, '?, ');
            elseif (task.isnominal(j))
                fprintf(dataFile, '%d, ', task.objects(i, j));
            else
                fprintf(dataFile, '%.5f, ', task.objects(i, j));
            end
        end
        
        fprintf(dataFile, '%d.\n', task.target(i));
    end
    
    fclose(dataFile);
end
