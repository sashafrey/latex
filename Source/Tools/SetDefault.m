function params = SetDefault(params, name, value)
    if (~isfield(params, name))
        params.(name) = value;
    end
end