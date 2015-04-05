function task = ConvertESokolovFormatToTask(X, Y)
    classes = unique(Y);
    Check(length(classes) == 2);
    Check(classes(1) == -1);
    Check(classes(2) == 1);
    
	Y(Y == 1) = 2;
    Y(Y == -1) = 1;
    task = struct('nItems', size(X, 1), ...
                  'nFeatures', size(X, 2), ...
                  'nClasses', 2, ...
                  'target', Y, ...
                  'objects', X, ...
                  'isnominal', false(size(X, 2), 1), ...
                  'name', 'noname');
end

