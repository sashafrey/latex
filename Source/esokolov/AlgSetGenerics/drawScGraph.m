function h = drawScGraph(A)
    h = maximizeFigure(figure);
    hold on;
    coords = zeros(size(A, 1), 2);
    algCnt = zeros(size(A, 2) + 1, 1);
    start = zeros(size(A, 2) + 1, 1);
    
    % считаем число вершин в каждом слое
    for i = 1:size(A, 1)
        m = sum(A(i, :));
        algCnt(m + 1) = algCnt(m + 1) + 1;
    end
    % где должен быть центр слоя, чтобы все было более-менее симметрично?
    center = floor(max(algCnt) / 2);
    for i = 1:length(start)
        start(i) = center - floor(algCnt(i) / 2);
    end
    
    % рисуем вершины
    algCnt = zeros(size(A, 2) + 1, 1);
    for i = 1:size(A, 1)
        m = sum(A(i, :));
        coords(i, 2) = m;
        coords(i, 1) = start(m + 1) + algCnt(m + 1);
        algCnt(m + 1) = algCnt(m + 1) + 1;
        %plot(coords(i, 1), coords(i, 2), '.r', 'MarkerSize', 15);
    end
    % рисуем ребра
    for i = 1:size(A, 1)
        for j = 1:size(A, 1)
            if (sum(abs(A(j, :) - A(i, :))) == 1 && sum(A(j, :) - A(i, :)) == 1)
                plot([coords(i, 1) coords(j, 1)], [coords(i, 2) coords(j, 2)], '-b');
            end
        end
    end
    for i = 1:size(A, 1)
        plot(coords(i, 1), coords(i, 2), '.r', 'MarkerSize', 15);
    end
    hold off;
end