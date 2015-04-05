%% линейно разделимая выборка
X = [1, 1;
     1, 2;
     1, 3;
     1, 4;
     1, 5;
     1, 6;
     1, 7;
     1, 8;
     2, 1;
     2, 2;
     2, 3;
     2, 4;
     2, 5;
     2, 6;
     2, 7;
     2, 8 ];
%X = X + randn(size(X)) / 1000;
X(1:2:end, 1) = X(1:2:end, 1) - 0.01;
X(1:4:end, 1) = X(1:4:end, 1) - 0.01;
Y = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1];

L = 16;
ell = 8;

A = getLinearAlgorithmsSet(X, Y);
scProfile = calculateScProfile(A, L);

eps = 0:0.005:1;
bounds_sep = zeros(size(eps));
for eps_idx = 1:length(eps)
    eps_curr = eps(eps_idx);
    bounds_sep(eps_idx) = calcScBoundFromScProfile(L, ell, eps_curr, scProfile);
end

h = plotSample(X, Y);
saveas(h, './esokolov/LinearClassifiers_Pics/SepSample', 'eps2c');
close(h);

h = drawScGraph(A);
saveas(h, './esokolov/LinearClassifiers_Pics/SepSampleGraph', 'eps2c');
close(h);

%% линейно неразделимая выборка
Y = [-1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1];

L = 16;
ell = 8;

A = getLinearAlgorithmsSet(X, Y);
scProfile = calculateScProfile(A, L);

eps = 0:0.005:1;
bounds_unsep = zeros(size(eps));
for eps_idx = 1:length(eps)
    eps_curr = eps(eps_idx);
    bounds_unsep(eps_idx) = calcScBoundFromScProfile(L, ell, eps_curr, scProfile);
end

h = plotSample(X, Y);
saveas(h, './esokolov/LinearClassifiers_Pics/UnsepSample', 'eps2c');
close(h);

h = drawScGraph(A);
saveas(h, './esokolov/LinearClassifiers_Pics/UnsepSampleGraph', 'eps2c');
close(h);

%%
h = maximizeFigure(figure);
hold on;
plot(eps, bounds_sep, 'b', 'LineWidth', 2);
plot(eps, bounds_unsep, 'r', 'LineWidth', 2);
plot([0, 1], [1, 1], '--k', 'LineWidth', 2);
grid on;
legend('Seperable', 'Unseparable');
saveas(h, './esokolov/LinearClassifiers_Pics/SepAndUnsepBounds', 'eps2c');
close(h);
