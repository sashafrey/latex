function alg = convertLinearAlgToStructure(w, X, Y)
% Для заданного линейного классификатора с весами w формирует структуру,
% описанную в initLinearAlgStructure().
% Попутно линейный классификатор w преобразуется в классификатор с таким же
% вектором ошибок, но проходящий через d точек выборки.
%
% Предполагается, что последний признак в X --- константный

    % w должен быть столбцом
    if size(w, 1) == 1
        w = w';
    end

    L = size(X, 1);
    d = size(X, 2) - 1; % константный признак не считаем за признак

    w = w ./ abs(w(d + 1)); % вес при константе должен быть равным единице по модулю

    currDescr = zeros(1, d);  % на пересечении каких гиперплоскостей лежит currVertex?
    currVertex = w;
    A = eye(d + 1);
    b = w;
    
    % упираем гиперплоскость в d объектов
    for i = 1:d
        b(i) = 0;
        % ищем гиперплоскость, находящуюся ближе всего к текущей точке
        bestDist = +Inf;
        bestIdx = 0;
        for j = 1:L
            if (ismember(j, currDescr))
                continue;
            end
            A(i, :) = X(j, :);
            p = A \ b;
            if (norm(p - currVertex) < bestDist)
                bestDist = norm(p - currVertex);
                bestIdx = j;
            end
        end
        currDescr(i) = bestIdx;
        A(i, :) = X(bestIdx, :);
        currVertex = A \ b;
        b((i + 1):d) = currVertex((i + 1):d);
    end
    
    % считаем вектор ошибок
    currErrVect = (sign(X * currVertex) ~= Y);
    
    errVectInit = (sign(X * w) ~= Y);
    currErrVect(currDescr) = errVectInit(currDescr);
    
    if (sum(currErrVect) < sum(errVectInit))
        currErrVect = 1 - currErrVect;
        currErrVect(currDescr) = errVectInit(currDescr);
        currVertex = -currVertex;
    end
    
    % заполняем структуру
    alg = initLinearAlgStructure();
    alg.w = currVertex';
    alg.errVect = currErrVect;
    alg.errCnt = sum(alg.errVect);
    alg.descr = sort(currDescr);
    alg.lowerNeighsCnt = 0;
    alg.upperNeighsCnt = 0;
    alg.lowerNeighbours = [];
    alg.upperNeighbours = [];
end
