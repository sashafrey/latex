%Строит по двумерной выборке семейство конъюнкций вида [x1 <= a1][x2 <= a2]
% и визуализирует это семейство
function VisualizeConjunctionSet(sample, sampleClasses)

[L dim] = size(sample);
if (dim ~= 2)
    return
end
sampleClasses( sampleClasses == -1) = 0;
sortedSample = sort(sample, 1);

figure
hold on 
%x = zeros(1, 1);
algs = zeros(1, L);
x = zeros(0, 0);
y = zeros(0, 0);
numAlgs = 0;
%Перебираем все возможные пары (a1, a2)
for i = 1:L
    for j =1:L
        a = [sortedSample(i, 1) + 1e-7, sortedSample(j, 2) + 1e-7 ];
        
        alg = ( (sample(:, 1) <= a(1) ) & (sample(:, 2) <= a(2) ) )' ~= sampleClasses;
        flag = true;
        for n = 1:numAlgs
            if (alg == algs(n, :) )
                flag = false;
            end
        end
        
        if (flag)
            numAlgs = numAlgs + 1;
            algs(numAlgs, :) = alg;
            x(numAlgs) = a(1);
            y(numAlgs) = a(2);
            
            for  n = 1:numAlgs
                if ( sum( abs( algs(n, :) -  algs(numAlgs, :) ) ) == 1)
                     line([x(n) x(numAlgs)], [y(n) y(numAlgs)]);
                end
            end
        end
        
        %algs(numAlgs, :) = alg;
       
        %текущий алгоритм
        
    end
end



scatter(x, y, 10, 'g', 'filled');
scatter(sample(:, 1), sample(:, 2), 60, sampleClasses, 'p');

hold off

end