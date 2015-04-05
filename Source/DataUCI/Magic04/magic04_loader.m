function magic04_loader(file)
    name = 'magic04.data';
    if (~exist('file', 'var'))
        file = name;
    end
    
    X = dlmread(file, ',');
    X_full = X(:, 1:end-1);
    Y_full = X(:, end);
    save('magic04_data', 'X_full', 'Y_full');
end