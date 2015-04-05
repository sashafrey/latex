function string = ArrayToString(array, aligned)
    space = '  ';
    if ((nargin == 2) && aligned), space = sprintf(' \t'); end;

    if (ischar(array))
        string = array;
        return;
    end
    
    string = '';
    if (length(array) > 1) 
        string = '[ ';    
    end
   
    for i=1:length(array)        
        if (i~=1), string = sprintf('%s,%s', string, space ); end
        
        if iscell(array)
            value = array{i};
        else
            value = array(i);
        end
        
        if ischar(value)
            string = [string, value];
        elseif (length(value) > 1)
            string = [string, '?'];
        elseif (isinteger(value) || isfloat(value))
            string = [string, num2str(value)]; 
        elseif (islogical(value))
            if (value)
                string = [string, '1'];
            else
                string = [string, '0'];
            end
        else
            string = [string, '?'];
        end
    end
    
    if (length(array) > 1)
        string = sprintf('%s \t]', string);
    end    
end