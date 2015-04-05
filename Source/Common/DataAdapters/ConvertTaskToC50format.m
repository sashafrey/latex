function ConvertTaskToC50format( task, taskName, outputFolder)
    % this converts task to the format acceptable by SEE (implementation of    
    % Quinlan's C5.0; http://www.rulequest.com/see5-win.html
    
    ConvertTaskToC50NamesFile(task, taskName, outputFolder);
    ConvertTaskToC50DataFile(task, taskName, outputFolder);    
 end

%fields = fieldnames(tasks);
%for i=1:length(fields)
%    name = fields{i};
%    ConvertTaskToC50format( tasks.(name), name, 'C5.0')
%end