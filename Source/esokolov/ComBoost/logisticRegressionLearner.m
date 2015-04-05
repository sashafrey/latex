function w = logisticRegressionLearner(X, Y)
    % нужен вектор из ноликов и единичек в качестве ответов
    Y(Y == -1) = 0;

    w = glmfit(X, Y, 'binomial', 'link', 'logit');
    w = [w(2:end); w(1)];   % "константный признак" должен стоять последним
    w = w ./ sqrt(w'* w);
end