function PlotArrangementOfHyperplanes(X, useColors)
    if ~exist('useColors', 'var')
        useColors = false;
    end
    
    if useColors
        colors = {'b', 'r', 'g', 'm', 'c', 'k'};
    else
        colors = {'b'};
    end

    Check(size(X, 2) == 3);
    
    nPoints = 100000;
    
    figure;
    hold on;
    for i = 1:size(X, 1)
        a = X(i, 1);
        b = X(i, 2);
        c = X(i, 3);
        
        spanVect1 = [0, c, -b]';
        spanVect1 = spanVect1 / norm(spanVect1);
        
        spanVect2 = cross(spanVect1, [a, b, c]');
        spanVect2 = spanVect2 / norm(spanVect2);
        
        pointsX = zeros(nPoints, 1);
        pointsY = zeros(nPoints, 1);
        pointsZ = zeros(nPoints, 1);
        for j = 1:nPoints
            angle = ((j - 1) / nPoints) * 2 * pi;
            alpha = cos(angle);
            beta = sin(angle);
            point = alpha * spanVect1 + beta * spanVect2;
            
            pointsX(j) = point(1);
            pointsY(j) = point(2);
            pointsZ(j) = point(3);
        end
        
        mask = (pointsZ >= 0);
        idx = find(~mask, 1, 'first');
        pointsX = [pointsX(idx:end); pointsX(1:idx-1)];
        pointsY = [pointsY(idx:end); pointsY(1:idx-1)];
        pointsZ = [pointsZ(idx:end); pointsZ(1:idx-1)];
        mask = [mask(idx:end); mask(1:idx-1)];
        pointsX(~mask) = [];
        pointsY(~mask) = [];
        pointsZ(~mask) = [];
        
        currColor = colors{1 + mod(i, length(colors))};
        plot3(pointsX, pointsY, pointsZ, currColor);
    end
    
    t = 1:100000;
    t = (t / 100000) * 2 * pi;
    plot(cos(t), sin(t), 'k', 'LineWidth', 2);
end