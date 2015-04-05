%%
% [X, Y] = generateNormalSample(200, 2, 1);
% save('./LinearClassifiers_Data/EqClassPic_data.mat', 'X', 'Y');

load('./LinearClassifiers_Data/EqClassPic_data.mat', 'X', 'Y');

%%
plotSample(X, Y);
hold on;

%%
plot([-4, 4], [4, -4], 'k', 'LineWidth', 2);
plot([-4, 4], [2, -3], 'k', 'LineWidth', 2);
plot([-1, 1], [4, -4], 'k', 'LineWidth', 2);

%%
saveas(gcf, './LinearClassifiers_Pics/pic_lin_eqClass', 'eps2c');