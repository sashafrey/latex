function w = logisticRegressionLearner(X, Y)
    % ����� ������ �� ������� � �������� � �������� �������
    Y(Y == -1) = 0;

    w = glmfit(X, Y, 'binomial', 'link', 'logit');
    w = [w(2:end); w(1)];   % "����������� �������" ������ ������ ���������
    w = w ./ sqrt(w'* w);
end