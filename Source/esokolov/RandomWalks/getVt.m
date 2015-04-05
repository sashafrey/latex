function Vt = getVt(X, Y, iterCnt, maxLevel, display)
% Функция для оценивания V_t --- числа алгоритмов в нижних maxLevel
% слоях графа расслоения-связности линейных классификаторов.

    L = size(X, 1);
    d = size(X, 2) - 1;
    algsCnt = computeNumberOfLinearAlgs(L, d);
    
    if ~exist('display', 'var')
        display = false;
    end
        
    lowerCnt = 0;
    
    for iter = 1:iterCnt
        if (display && mod(iter, 100) == 0)
            fprintf('%d\n', iter);
        end
        
        descr = randsample(L, d);
        
        A = X(descr, :);
        A(d + 1, 1:(d + 1)) = 0;
        A(d + 1, d + 1) = 1;
        b = zeros(d + 1, 1);
        if (rand(1) > 0.5)
            b(d + 1) = 1;
        else
            b(d + 1) = -1;
        end
        w = A \ b;
%         errVect = zeros(1, L);
%         for i = 1:L
%             errVect(i) = (sign(X(i, :) * w) ~= Y(i));
%         end
        errVect = (sign(X * w) ~= Y);
        errVect(descr) = 0.5 + sign(rand(1, d) - 0.5) / 2;
        
        errCnt = sum(errVect);
        if (errCnt <= maxLevel)
            lowerCnt = lowerCnt + 1;
        end
    end
    
    Vt = algsCnt * (lowerCnt / iterCnt);
end
