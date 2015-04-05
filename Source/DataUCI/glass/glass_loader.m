function glass_loader(file)
    name = 'glass.data';
    if (~exist('file', 'var'))
        file = name;
    end
    
    X = dlmread(file, ',');
    X_full = X(:, 2:end-1);
    Y_full = 2*double(ismember(X(:, end), [1 2 3 4])) - 1;
    save('glass_data', 'X_full', 'Y_full');
end