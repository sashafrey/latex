function alg = initLinearAlgStructure(w, errVect, errCnt, descr, ...
    lowerNeighsCnt, upperNeighsCnt, lowerNeighbours, upperNeighbours)
% ¬озвращает структуру, описывающую линейный классификтор

    alg = struct('w', [], ...   % веса, соответствующие данному классификатору
                 'errVect', [], ... % вектор ошибок
                 'errCnt', [], ...  % число ошибок
                 'descr', [], ...   % номера объектов выборки, 
                 ...                % через которые проходит гиперплоскость
                 'lowerNeighsCnt', [], ... % число нижних соседей
                 'upperNeighsCnt', [], ...  % число верхних соседей
                 'lowerNeighbours', [], ... % индексы нижних соседей
                 ...                        % (если алгоритмы хран€тс€ в одном массиве)
                 'upperNeighbours', []... % индексы верхних соседей
                 );
             
    if nargin > 0
        alg.w = w;
        alg.errVect = errVect;
        alg.errCnt = errCnt;
        alg.descr = descr;
        alg.lowerNeighsCnt = lowerNeighsCnt;
        alg.upperNeighsCnt = upperNeighsCnt;
        alg.lowerNeighbours = lowerNeighbours;
        alg.upperNeighbours = upperNeighbours;
    end
end
