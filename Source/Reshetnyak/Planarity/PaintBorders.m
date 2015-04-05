function  PaintBorders(sample, sampleClasses)
%PAINTBORDERS ѕо двумерной выборке sample рисует обасти значений
%параметров, соответствующих различным  линейным алгоритмам.
%   Detailed explanation goes here
    sampleSize = size(sample, 1);
%»щем все точки пересечени€ пр€мых
    minX = 0; maxX = 0; minY = 0; maxY = 0;
    intersectionPoint = zeros(sampleSize * (sampleSize - 1) /2, 2)
    cnt  = 0;
    for i = 1:sampleSize
        for j = i+1:sampleSize
            cnt = cnt + 1;
            intersectionPoint(cnt, :) =  sample([i, j], :) \ [1 ; 1];
            minX = min(intersectionPoint(cnt,1), minX);
            maxX = max(intersectionPoint(cnt,1), maxX);
            minY = min(intersectionPoint(cnt,2), minY);
            maxY = max(intersectionPoint(cnt,2), maxY);
        end
    end
    
    minX = minX - 1;
    minY = minY - 1;
    maxX = maxX + 1;
    maxY = maxY + 1;
    hold on
    for i = 1:sampleSize
        if sample(i, 2) == 0
            line([1 / sample(i, 1) 1 / sample(i, 1)], [minY maxY]);
        else
            line( [minX maxX], [1 - sample(i, 1) * minX, 1 - sample(i, 1) * maxX] / sample(i, 2));
        end
    end
    xlim([minX maxX]);
    ylim([minY maxY]);
    
    
    minDist = 1e9;
    for i = 1:size(intersectionPoint, 1)
        for j = i+1: size(intersectionPoint, 1)
            minDist = min(minDist, sqrt( sum((intersectionPoint(i, :) - intersectionPoint(j, :)).^2) ) );
        end
    end
    minDist
    
    
    %plotting linear set
    algs = zeros(200000, sampleSize);
    coords = zeros(200000, 2);
    sampleClasses(sampleClasses == 0) = -1;
   
    pos = 1;
    x = minX + 0.5;
    while (x < maxX - 0.5)
        y = minY + 0.5;
        while (y < maxY - 0.5)
            coords(pos, :) = [x y];
            algs(pos, :) = ( ((sample * [x;y] - 1) .* sampleClasses') < 0);
            pos = pos + 1;
            algs(pos, :) = ~algs(pos - 1, :);
            coords(pos, :) = -[x y];
            pos = pos + 1;
            y = y + 0.5 *minDist;
        end
        x 
        'bzz'
        x = x + 0.5 * minDist;
    end
    [algs, ind] = sortrows(algs);
    coords = coords(ind, :);
    
    uniqAlgs = false(sampleSize * (sampleSize - 1) + 2, sampleSize);
    uniqCoords = zeros(sampleSize * (sampleSize - 1) + 2, 2);
    n = 1;
    uniqAlgs(1, :) =  algs(1, :);
    uniqCoords(1, :) = coords(1, :);
	for i = 2:size(algs, 1)
		if (algs(i, :) == algs(i-1, :) )
			continue;
		end
		n = n + 1;
		uniqAlgs(n, :) = algs(i, :);
        uniqCoords(n, :) = coords(i, :);
    end
    scatter( uniqCoords(1:n,1),uniqCoords(1:n,2), 10, 'r', 'filled');
    hold off
    PaintAlgorithmsFamily(uniqAlgs(1:n, :));
    
end
