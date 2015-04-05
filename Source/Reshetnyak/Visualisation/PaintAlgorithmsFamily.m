%Рисует семейство алгоритмов
%algs - матрица L x n из 0 и 1
%numLevels - число нихних уровней графа, которые будут нарисованы
%Алгоритм изображается точкой на рисунке
%Координата y точки равна числу ошибок алгоритма
%Если isGeneralized = false, то рёбрами соединяются алгоритмы, отличающиеся на одном объекте
%Если isGeneralized = true, то граф - граф Хассе

function PaintAlgorithmsFamily(algs, numLevels, isGeneralized)
    
    function PlotArc(xx, yy)
        if xx(2) ~= xx(1)
            a = (yy(2) - yy(1)) / (xx(2) - xx(1))^2;
            b = -2 * a * xx(1);
            c = yy(1) + a * xx(1)^2;
            p = [xx(1) : 0.05 : xx(2)];
            line(p, a * p .^2 + b * p + c)
        else
            a =  2 / (yy(2) - yy(1))^2;
            b = -a * (yy(1) + yy(2));
            c = 0.25 * a * (yy(1) + yy(2))^2 + xx(1) - 0.5;
            p = [yy(1) : 0.05 : yy(2)];
            line(a * p .^2 + b * p + c, p);
        end
    end

	[numAlgs sampleSize] = size(algs);
    totalError = sum(algs, 2);
    [totalError ind] = sort(totalError);
    algs = algs(ind, :);
    minError = min(totalError);
    if nargin < 3
        isGeneralized = false
    end
    if nargin < 2
        numLevels = sampleSize + 1;
    end
    numLevels = min(numLevels, max(totalError) - minError + 1);
    
  
    if isGeneralized
        [graph, levels] = BuildHasseGraph(algs, minError + numLevels);
    else
        [graph, levels] = BuildFamilyGraph(algs, minError + numLevels);
    end
    x = zeros(1, numel(graph));
    y = zeros(1, numel(graph));
	
    for errorCount = minError : minError + numLevels - 1 
        %k = ceil(-levels(m + 1)/2);
		k = 1;
        start = 1;
        if errorCount > 0
            start = levels(errorCount);
        end
        for i = start : levels(errorCount + 1) - 1;
            y(i) = errorCount;
            x(i) = k;
            k = k + 1;
        end            
    end
    
    figure
    hold on
    for n = 1:numel(graph)
        for v = 1:numel(graph{n})
            if isGeneralized && (abs(y(n) - y(graph{n}(v))) > 1)
                PlotArc([x(n) x(graph{n}(v))], [y(n) y(graph{n}(v))]);
            else
                line([x(n) x(graph{n}(v))], [y(n) y(graph{n}(v))]);
            end
                
        end
    end
    scatter(x, y, 30, 'r', 'filled');
	%axis([min(x) max(x) + 5 min(y) max(y) + 5])
    hold off
    
end