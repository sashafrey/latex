function pageblocks_loader(file)
    name = 'page-blocks.data';
    if (~exist('file', 'var'))
        file = name;
    end
    
    X = dlmread(file, ';');
    X_full = X(:, 1:end-1);
    Y_full = 2*double(X(:, end) == 1) - 1; % First class = letters A...M
    save('pageblocks_data', 'X_full', 'Y_full');
end