function ConvertTaskToC50NamesFile( task, taskName, outputFolder )
    % this converts task to the format acceptable by SEE (implementation of    
    % Quinlan's C5.0; http://www.rulequest.com/see5-win.html
    
    if (~exist('outputFolder', 'var'))
        outputFolder = '.';
    end
    
    if (~exist(outputFolder,'dir'))
        mkdir(outputFolder);
    end
    
    namesFile = fopen(fullfile(outputFolder, sprintf('%s.names', taskName)), 'w');
    fprintf(namesFile, 'target\t| the target attribute\n\n');
    for i=1:task.nFeatures
        if (task.isnominal(i))
            fprintf(namesFile, 'feature%i:\t', i);
            values = unique(task.objects(:, i));
            values = values(~isnan(values));

            first = true;
            for value = values'
                if (~first)
                    fprintf(namesFile, ', ');
                end
                first = false;
                fprintf(namesFile, '%d', value);                
            end
            
            fprintf(namesFile, '.\n');
        else
            fprintf(namesFile, 'feature%i:\tcontinuous.\n', i);
        end
    end
    
    fprintf(namesFile, '\ntarget:\t');
    for i=1:task.nClasses
        fprintf(namesFile, '%d', i);
        if (i ~= task.nClasses)
            fprintf(namesFile, ', ');
        else
            fprintf(namesFile, '.\n');
        end
    end
    
    fclose(namesFile);
end
