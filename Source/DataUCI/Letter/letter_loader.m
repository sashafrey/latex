function letter_loader(file)
    name = 'letter-recognition.data';
    if (~exist('file', 'var'))
        file = name;
    end
    
    X = dlmread(file, ',');
    X_full = X(:, 2:end);
    Y_full = 2*double(X(:, 1) <= 13) - 1; % First class = letters A...M
    save('letter-recognition_data', 'X_full', 'Y_full');
end