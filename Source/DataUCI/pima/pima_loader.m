function pima_loader(file)
    name = 'pima-indians-diabetes.data';
    if (~exist('file', 'var'))
        file = name;
    end
    
    X = dlmread(file, ',');
    X_full = X(:, 1:end-1);
    Y_full = 2*X(:, end) - 1; 
    save('pima_data', 'X_full', 'Y_full');
end