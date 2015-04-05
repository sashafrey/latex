function optdigits_loader(file)
    name = 'uci-20070111-optdigits.csv';
    if (~exist('file', 'var'))
        file = name;
    end
    
    X = dlmread(file, ',');
    X_full = X(:, 1:end-1);
    Y_full = 2 * double(X(:, end) < 5) - 1;
    save('optdigits_data', 'X_full', 'Y_full');
end