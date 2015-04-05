%%
clear;

%% генерируем модельную разделимую выборку (каждый класс - гауссиана)
L = 500;
d = 100;
sigma = 1;
[X, Y, errCnt] = generateNormalSample(L, d, 10);

%% вычисляем PAC-Bayes оценки для разного числа шумовых признаков
noiseFeaturesCnt = 0:10:500;
pacBayesDdBounds = nan(length(noiseFeaturesCnt), 1);
pacBayesDiBounds = nan(length(noiseFeaturesCnt), 1);
cvBounds = nan(length(noiseFeaturesCnt), 1);

parfor noiseCntIdx = 1:length(noiseFeaturesCnt)
    warning off;
    fprintf('%d\n', noiseCntIdx);
    
    currNoiseFeaturesCnt = noiseFeaturesCnt(noiseCntIdx);
    
    noise = randn(L, currNoiseFeaturesCnt);
    X_curr = [X, noise];
    
    % обучаем логистическую регрессию и считаем отступы
    w = logisticRegressionLearner(X_curr, Y);

    X_curr_1 = [X_curr, ones(L, 1)];
    margins = Y .* (X_curr_1 * w) ./ ...
        sqrt(sum(X_curr_1 .^ 2, 2));

    %
    curr_d = size(X_curr, 2);
    pacBayesDdBounds(noiseCntIdx) = DDmargin(margins, curr_d);
    pacBayesDiBounds(noiseCntIdx) = DImargin(margins);
    
    cvBounds(noiseCntIdx) = getCvBound_Linear(X_curr, Y, 10, ...
        @logisticRegressionLearner);
end

%%
h = maximizeFigure(figure);
hold on;
plot(noiseFeaturesCnt, cvBounds, 'b', 'LineWidth', 2);
plot(noiseFeaturesCnt, pacBayesDiBounds, 'r', 'LineWidth', 2);
plot(noiseFeaturesCnt, pacBayesDdBounds, 'g', 'LineWidth', 2);
grid on;
legend('CV', 'PAC-Bayes DI', 'PAC-Bayes DD');
xlabel('Noise features count');
