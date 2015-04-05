function p = params(varargin)
    % PARAMS converts variable argument list to struct.
    %
    % Example:
    % PARAMS('size', 5, 'property', 'some text') outputs struct with two 
    % fields: field 'size' with value 5, and field 'property' with value
    % 'some text'.
    
	nargs = length(varargin);
    Check(rem(nargs, 2) == 0, 'Number of arguments must be even.');
    keys = varargin(1:2:nargs);
    vals = varargin(2:2:nargs);
    for i = 1:length(keys)
        p.(keys{i}) = vals{i};
    end    
end