function [alg, neighbours] = getNeighboursLC_rays(alg, X, Y, raysCnt)
% Возвращает для алгоритма alg всех соседей, найденных методом случайных
% направлений.

    [N, d] = size(X);
    eps = 1e-6;
    
    neighbours = StructVectorCreate(initLinearAlgSimpleStructure());
    
    dir_all = getRandomDirections(raysCnt, d);
    
    for currRayNum = 1:raysCnt
        % генерируем направление луча
        dir = dir_all(currRayNum, :)';
        
        % пересечения луча с гиперплоскостями
        intersections = inf(N, 1);
        
        % TODO: векторизовать цикл
        for currHyperplane = 1:N
            w_curr = X(currHyperplane, :)';
            % если луч параллелен гиперплоскости, то пропускаем ее
            if abs(w_curr' * dir) < eps
                continue;
            end
            
            intersections(currHyperplane) = ...
                - (w_curr' * alg.w) / (w_curr' * dir);
        end
        
        % а не направили ли мы луч в пустоту?
        % если да, то разворачиваем его
        if sum(intersections > 0) <= 1
            dir = -dir;
            intersections = -intersections;
        end
        
        % если коэффициент оказался меньше нуля, то луч не пересекает
        % гиперплоскость; выбрасываем такие пересечения
        intersections(intersections <= 0) = Inf;
        
        intersect_sorted = sort(intersections);
        coef_new = (intersect_sorted(1) + intersect_sorted(2)) / 2;
        w_new = alg.w + coef_new * dir;
        
        alg_new = convertLinearAlgToSimpleStructure(w_new, X, Y);
        
        % может, такой алгоритм уже есть в neighbours?
        is_new = true;
        for neighIdx = 1:StructVectorLength(neighbours)
            if isEqualLC(neighbours.Data(neighIdx), alg_new)
                is_new = false;
                break;
            end
        end
        
        % если такого мы еще не видели, то срочно добавляем!
        if is_new
            neighbours = StructVectorAdd(neighbours, alg_new);
            
            if alg_new.errCnt < alg.errCnt
                alg.lowerNeighsCnt = alg.lowerNeighsCnt + 1;
            else
                alg.upperNeighsCnt = alg.upperNeighsCnt + 1;
            end
        end
    end    
end
