%%
load('./esokolov/Experiments/RW/unsep200_rw_res.mat');

%% клевые картинки
rw_pics_path = './esokolov/Experiments/RW';
colors = { [0 0 0] / 255;
           [139 137 137] / 255;
           [49 255 79] / 255;
           [225 25 152] / 255;
           [72 61 239] / 255;
           [0 206 209] / 255;
           [34 139 34] / 255;
           [178 34 0] / 255;
           [148 0 255] / 255;
           [255 0 0] / 255;
           [255 165 0] / 255};
eps_idx = 5:20;
       
%% оценки для всех eps
legends = {};
colorIdx = 1;

h = maximizeFigure(figure);
plot(eps_arr_rw(eps_idx), bounds_true(eps_idx), 'LineWidth', 3, 'Color', colors{colorIdx});
colorIdx = colorIdx + 1;
hold on;
legends = [legends; 'True bound'];

for rwIdx = 1:length(random_walkers.walker_name)
    plot(eps_arr_rw(eps_idx), bounds_simple_all_eps(rwIdx, eps_idx), 'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, simple estimate', random_walkers.walker_name{rwIdx})];
end
for rwIdx = 1:length(random_walkers.walker_name)
    plot(eps_arr_rw(eps_idx), bounds_layered_all_eps(rwIdx, eps_idx), 'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, layered estimate', random_walkers.walker_name{rwIdx})];
end

set(gca, 'XLim', [eps_arr_rw(eps_idx(1)), eps_arr_rw(eps_idx(end))]);
grid on;
legend(legends, 'Location', 'NorthEast');
xlabel('epsilon');
ylabel('Q_\epsilon');

saveas(h, ...
    sprintf('%s/eps_all', rw_pics_path), ...
    'eps2c');

%% как предыдущая картинка, но теперь отклонения
legends = {};
colorIdx = 2;

h = maximizeFigure(figure);
hold on;

for rwIdx = 1:length(random_walkers.walker_name)
    plot(eps_arr_rw(eps_idx), bounds_simple_all_eps(rwIdx, eps_idx) - bounds_true(eps_idx)', ...
        'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, simple estimate', random_walkers.walker_name{rwIdx})];
end
for rwIdx = 1:length(random_walkers.walker_name)
    plot(eps_arr_rw(eps_idx), bounds_layered_all_eps(rwIdx, eps_idx) - bounds_true(eps_idx)', ...
        'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, layered estimate', random_walkers.walker_name{rwIdx})];
end

set(gca, 'XLim', [eps_arr_rw(eps_idx(1)), eps_arr_rw(eps_idx(end))]);
grid on;
legend(legends, 'Location', 'NorthEast');
xlabel('epsilon');
ylabel('Q_\epsilon');

saveas(h, ...
    sprintf('%s/eps_reminders_all', rw_pics_path), ...
    'eps2c');

%% оценки для всех длин выборок
legends = {};
colorIdx = 1;

h = maximizeFigure(figure);
plot([lengths_rw(1), lengths_rw(end)], ...
    [bounds_true(abs(eps_arr_rw - eps_rw) < 1e-6), bounds_true(abs(eps_arr_rw - eps_rw) < 1e-6)], ...
    'LineWidth', 3, 'Color', colors{colorIdx});
colorIdx = colorIdx + 1;
hold on;
legends = [legends; 'True bound'];

for rwIdx = 1:length(random_walkers.walker_name)
    plot(lengths_rw, bounds_simple_all_len(rwIdx, :), 'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, simple estimate', random_walkers.walker_name{rwIdx})];
end
for rwIdx = 1:length(random_walkers.walker_name)
    plot(lengths_rw, bounds_layered_all_len(rwIdx, :), 'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, layered estimate', random_walkers.walker_name{rwIdx})];
end

grid on;
legend(legends, 'Location', 'NorthEast');
xlabel('epsilon');
ylabel('Q_\epsilon');

saveas(h, ...
    sprintf('%s/lengths_all', rw_pics_path), ...
    'eps2c');

%% как предыдущая картинка, но теперь отклонения
legends = {};
colorIdx = 2;

h = maximizeFigure(figure);
hold on;
legends = [legends; 'True bound'];

for rwIdx = 1:length(random_walkers.walker_name)
    plot(lengths_rw, bounds_simple_all_len(rwIdx, :) - bounds_true(eps_arr_rw == eps_rw), ...
        'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, simple estimate', random_walkers.walker_name{rwIdx})];
end
for rwIdx = 1:length(random_walkers.walker_name)
    plot(lengths_rw, bounds_layered_all_len(rwIdx, :) - bounds_true(eps_arr_rw == eps_rw), ...
        'LineWidth', 2, 'Color', colors{colorIdx});
    colorIdx = colorIdx + 1;
    legends = [legends; ...
        sprintf('%s, layered estimate', random_walkers.walker_name{rwIdx})];
end

grid on;
legend(legends, 'Location', 'NorthEast');
xlabel('epsilon');
ylabel('Q_\epsilon');

saveas(h, ...
    sprintf('%s/lengths_reminders_all', rw_pics_path), ...
    'eps2c');