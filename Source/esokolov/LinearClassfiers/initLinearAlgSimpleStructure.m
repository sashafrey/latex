function alg = initLinearAlgSimpleStructure(w, errVect, errCnt, ...
    lowerNeighsCnt, upperNeighsCnt, lowerNeighbours, upperNeighbours)
% Возвращает структуру, описывающую линейный классификтор

    alg = struct('w', [], ...   % веса, соответствующие данному классификатору
                 'errVect', [], ... % вектор ошибок
                 'errCnt', [], ...  % число ошибок
                 'lowerNeighsCnt', [], ... % число нижних соседей
                 'upperNeighsCnt', [], ...  % число верхних соседей
                 'lowerNeighbours', [], ... % индексы нижних соседей
                 ...                        % (если алгоритмы хранятся в одном массиве)
                 'upperNeighbours', []... % индексы верхних соседей
                 );
             
    if nargin > 0
        alg.w = w;
        alg.errVect = errVect;
        alg.errCnt = errCnt;
        alg.lowerNeighsCnt = lowerNeighsCnt;
        alg.upperNeighsCnt = upperNeighsCnt;
        alg.lowerNeighbours = lowerNeighbours;
        alg.upperNeighbours = upperNeighbours;
    end
end
