function HandleException(ex, Message)
    if (nargin == 1)
        Message = 'Exception occurred';
    end
    
    if ~verLessThan('matlab', '7.7.0')
        ErrStr = getReport(ex, 'extended');      
    else
        ErrStr = sprintf('%s. File: %s, Function: %s, Line: %d',ex.message,ex.stack(1).file, ex.stack(1).name, ex.stack(1).line);
    end
    logf = fopen('log.txt', 'at');
    fprintf(logf, '%s: %s. %s\n', datestr(now), Message, ErrStr);
    fclose(logf);
    fprintf('%s: %s. %s\n', datestr(now), Message, ErrStr);
end