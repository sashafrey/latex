L = 8;
ell = 4;

%% линейно разделимая выборка
[X, Y] = generateNormalSample(L, 2, 1);

A = getLinearAlgorithmsSet(X, Y);
scProfile = calculateScProfile(A, L);

eps = 0:0.005:1;
bounds_sep = zeros(size(eps));
for eps_idx = 1:length(eps)
    eps_curr = eps(eps_idx);
    bounds_sep(eps_idx) = calcScBoundFromScProfile(L, ell, eps_curr, scProfile);
end

h = plotSample(X, Y);
saveas(h, './esokolov/LinearClassifiers_Pics/SepSample_1', 'eps2c');
close(h);

h = drawScGraph(A);
axis off;
saveas(h, './esokolov/LinearClassifiers_Pics/SepSampleGraph_1', 'eps2c');
close(h);

%% линейно неразделимая выборка
[X, Y] = generateNormalSample(L, 2, 50);

A = getLinearAlgorithmsSet(X, Y);
scProfile = calculateScProfile(A, L);

eps = 0:0.005:1;
bounds_unsep = zeros(size(eps));
for eps_idx = 1:length(eps)
    eps_curr = eps(eps_idx);
    bounds_unsep(eps_idx) = calcScBoundFromScProfile(L, ell, eps_curr, scProfile);
end

h = plotSample(X, Y);
saveas(h, './esokolov/LinearClassifiers_Pics/UnsepSample_1', 'eps2c');
close(h);

h = drawScGraph(A);
axis off;
saveas(h, './esokolov/LinearClassifiers_Pics/UnsepSampleGraph_1', 'eps2c');
close(h);

%%
h = maximizeFigure(figure);
hold on;
plot(eps, bounds_sep, 'b', 'LineWidth', 2);
plot(eps, bounds_unsep, 'r', 'LineWidth', 2);
plot([0, 1], [1, 1], '--k', 'LineWidth', 2);
grid on;
legend('Seperable', 'Unseparable');
saveas(h, './esokolov/LinearClassifiers_Pics/SepAndUnsepBounds_1', 'eps2c');
close(h);
