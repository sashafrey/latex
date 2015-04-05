function Check(expression, message)
    if (expression == 0)
        if (nargin == 1), message = 'error'; end;
        disp(['Check failed: ', message]);
        s = dbstack;
        for i=1:length(s)
            disp(sprintf('\tfile %s, line %d', s(i).file, s(i).line));
        end
        
        disp(['Check Failed, with message: ', message]);
        ME = MException('Cash4Cast:Check', message);
        throw(ME);
    end
end