function h = plotSample(X, Y)
    h = figure;
    hold on;
    plot(X(Y == -1, 1), X(Y == -1, 2), '*r', 'MarkerSize', 6);
    plot(X(Y == 1, 1), X(Y == 1, 2), 'ob', 'MarkerSize', 6);
    hold off;
end