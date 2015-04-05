%%
% в памяти должна лежать переменная cvLog

%%
iFunc = 1;
iBaseAlg = 1;

%%
corrSample = [];
ndcgSample = [];
for iCvFold = 1:5
    for iFeatureSelectionStep = 2:5
        %for iBaseAlg = 1:10
            fprintf('.');
            cmp = CompareGradesAndTestErrors(cvLog, iFunc, iCvFold, iBaseAlg, ...
                iFeatureSelectionStep);
            currCorr = corr(cmp(:, 1), cmp(:, 2), 'type', 'Kendall');
            corrSample(end + 1) = currCorr;
            
            [~, ix] = sort(cmp(:, 2));
            cmp = cmp(ix, :);
            cmp(:, 1) = size(cmp, 1) - cmp(:, 1) + 1;
            ndcg = cmp(1, 1) + sum(cmp(2:end, 1) ./ log2(cmp(2:end, 2)));
            a = size(cmp, 1) - cmp(:, 2) + 1;
            ndcg = ndcg / (a(1) + sum(a(2:end) ./ log2(cmp(2:end, 2))));
            ndcgSample(end + 1) = ndcg;
        %end
    end
end
fprintf('\n');
corrSample = corrSample';
ndcgSample = ndcgSample';

%%
iFunc = 1;
iBaseAlg = 1;
iCvFold = 5;
iFeatureSelectionStep = 5;

b = [];
for i = 1:1000
    cmp = CompareGradesAndTestErrors(cvLog, iFunc, iCvFold, iBaseAlg, ...
                iFeatureSelectionStep);
    
    [~, ix] = sort(cmp(:, 2));
    cmp(:, 1) = cmp(randperm(size(cmp, 1)), 1);
    cmp = cmp(ix, :);
    cmp(:, 1) = size(cmp, 1) - cmp(:, 1) + 1;
    ndcg = cmp(1, 1) + sum(cmp(2:end, 1) ./ log2(cmp(2:end, 2)));
    a = size(cmp, 1) - cmp(:, 2) + 1;
    ndcg = ndcg / (a(1) + sum(a(2:end) ./ log2(cmp(2:end, 2))));
    b(end + 1) = ndcg;
end

%%
figNum = 7;

iFunc = 1;
iBaseAlg = 1;
iCvFold = 4;
iFeatureSelectionStep = 4;

cmp = CompareGradesAndTestErrors(cvLog, iFunc, iCvFold, iBaseAlg, ...
    iFeatureSelectionStep);

[~, ix] = sort(cmp(:, 2));
cmp = cmp(ix, :);
a = zeros(size(cmp, 1), 1);
for i = 1:length(a)
    a(i) = sum(cmp(1:i, 1) <= i);
end

%figure;
subplot(4, 2, figNum);
cla;
plot(a, 'LineWidth', 2);
hold on;
plot([1, length(a)], [1, length(a)], '-.k', 'LineWidth', 2);
%set(gca, 'XLim', [0, size(cmp, 1)+1]);
%set(gca, 'YLim', [0, size(cmp, 1)+1]);
grid on;
%saveas(gcf, sprintf('./esokolov/ComBoost_Experiments/Correlations/cv_%.2d_roc', figNum), ...
%    'fig');
%saveas(gcf, sprintf('./esokolov/ComBoost_Experiments/Correlations/cv_%.2d_roc', figNum), ...
%    'eps2c');

%figure;
subplot(4, 2, figNum + 1);
cla;
plot(cmp(:, 1), cmp(:, 2), '*', 'MarkerSize', 10);
set(gca, 'XLim', [0, size(cmp, 1)+1]);
xlabel('Test error');
ylabel('Sampled MCC');
%saveas(gcf, sprintf('./esokolov/ComBoost_Experiments/Correlations/cv_%.2d', figNum), ...
%    'fig');
%saveas(gcf, sprintf('./esokolov/ComBoost_Experiments/Correlations/cv_%.2d', figNum), ...
%    'eps2c');
