function grade = getSampleGrade_empirical(X, Y, baseClassifierLearner)
    L = size(X, 1);

    w = baseClassifierLearner(X, Y);
    
    X = [X, ones(L, 1)];
    Y_predicted = sign(X * w);
    
    grade = sum(Y ~= Y_predicted) / L;
end