function [algsVect, sourcesVect] = ...
    getNeighboursLC(algsVect, sourcesVect, currAlgNum, X, Y)
% Функция находит все вершины, соседние с algsVect(currAlgNum) в графе
% расслоения-связности для семейства линейных классификаторов над 
% выборкой X с метками Y
    L = size(X, 1);
    d = size(X, 2) - 1;
    
    alg = algsVect.Data(currAlgNum);
    lowerNeighbours = zeros(1, L);
    upperNeighbours = zeros(1, L);
    lowerNeighsCnt = 0;
    upperNeighsCnt = 0;
    
    % TODO: достаточно ли L?
    descrs = VectorCreate(alg.descr);
    vertices = VectorCreate(alg.w);
    
    i = 1;
    while (i <= descrs.Count)
        [descrs, vertices] = ...
            getIncidentVertices(descrs, vertices, i, alg, X, Y);
        i = i + 1;
    end
    
    for i = 1:descrs.Count
        newErrVect = alg.errVect;
        for k = 1:d
            newErrVect(descrs.Data(i, k)) = ~newErrVect(descrs.Data(i, k));
            newErrCnt = alg.errCnt;
            if (newErrVect(descrs.Data(i, k)) == 0)
                newErrCnt = newErrCnt - 1;
            else
                newErrCnt = newErrCnt + 1;
            end

            % findAlgorithm
            idx = -1;
            for z = 1:algsVect.Count
                if sum(algsVect.Data(z).errVect ~= newErrVect) == 0
                    idx = z;
                    break;
                end
            end

            % insertAlgorithm
            if (idx == -1)
                alg_insert = ...
                    initLinearAlgStructure(vertices.Data(i, :), ...
                        newErrVect, ...
                        newErrCnt, ...
                        descrs.Data(i, :), ...
                        0, 0, [], []);
                algsVect = StructVectorAdd(algsVect, alg_insert);
                idx = algsVect.Count;
            end

            if (newErrVect(descrs.Data(i, k)) == 0)
                lowerNeighsCnt = lowerNeighsCnt + 1;
                lowerNeighbours(lowerNeighsCnt) = idx;
            else
                upperNeighsCnt = upperNeighsCnt + 1;
                upperNeighbours(upperNeighsCnt) = idx;
            end
            newErrVect(descrs.Data(i, k)) = ~newErrVect(descrs.Data(i, k));
        end
    end
    
    lowerNeighbours = unique(lowerNeighbours(1:lowerNeighsCnt));
    lowerNeighsCnt = length(lowerNeighbours);
    upperNeighbours = unique(upperNeighbours(1:upperNeighsCnt));
    upperNeighsCnt = length(upperNeighbours);
    alg.lowerNeighsCnt = lowerNeighsCnt;
    alg.upperNeighsCnt = upperNeighsCnt;
    alg.lowerNeighbours = lowerNeighbours;
    alg.upperNeighbours = upperNeighbours;
    algsVect.Data(currAlgNum) = alg;
    
    if (alg.lowerNeighsCnt == 0)
        sourcesVect = VectorAdd(sourcesVect, currAlgNum);
    end
end

function [descrs, vertices] = getIncidentVertices(descrs, vertices, ...
    currDescrNum, alg, X, Y)
    L = size(X, 1);
    d = size(X, 2) - 1;
    
    currDescr = descrs.Data(currDescrNum, :);

    A = eye(d + 1);
    A(1:d, :) = X(currDescr, :);
    inverseA = inv(A);
    b = zeros(d + 1, 1);
    %b(d + 1) = alg.vertex(d + 1);
    for i = 1:d
        % надо придумать, как сделать здесь бинарный поиск!
        for s = [-1, 1]
            b(d + 1) = s;
            
            closestPoint = -ones(2, 1); % 1 - "left", 2 - "right"
            minDist = inf(2, 1);
            leftVect = zeros(1, d + 1);
            for j = setdiff(1:L, currDescr)
                %A(i, :) = X(j, :);
                %newVertex = (A \ b)';
                newVertex = recalculateInverse(inverseA, i, X(j, :), A(i, :)) * b;
                newVertex = newVertex';
                if (closestPoint(1) == -1)
                    closestPoint(1) = j;
                    minDist(1) = dist(vertices.Data(currDescrNum, :), newVertex);
                    leftVect = newVertex - vertices.Data(currDescrNum, :);
                else
                    if ((newVertex - vertices.Data(currDescrNum, :)) * leftVect' > 0)
                         % если эта точка по левую сторону
                        idx = 1;
                    else
                        idx = 2;
                    end
                    newDist = dist(vertices.Data(currDescrNum, :), newVertex);
                    if (newDist < minDist(idx))
                        minDist(idx) = newDist;
                        closestPoint(idx) = j;
                    end
                end
            end
            
            for j = 1:2
                if (closestPoint(j) == -1)
                    continue;
                end
                A(i, :) = X(closestPoint(j), :);
                newVertex = A \ b;
                newDescr = currDescr;
                newDescr(i) = closestPoint(j);
                newErrVect = (sign(X * newVertex) ~= Y);
                newErrVect(newDescr) = alg.errVect(newDescr);
                if sum(newErrVect ~= alg.errVect) == 0 
                    % нашли еще одну вершину ячейки
                    isNewDescr = 1;
                    sortedNewDescr = sort(newDescr);
                    for k = 1:descrs.Count
                        if sum(sortedNewDescr ~= descrs.Data(k, :)) == 0
                            isNewDescr = 0;
                            break;
                        end
                    end
                    if isNewDescr
                        descrs = VectorAdd(descrs, sortedNewDescr);
                        vertices = VectorAdd(vertices, newVertex');
                    end
                end
            end
            A(i, :) = X(currDescr(i), :);
        end
    end
end

function d = dist(x, y)
    d = sum((x - y) .^ 2);
end

function newInverse = recalculateInverse(inverseA, newRowNum, newRow, oldRow)
    n = size(inverseA, 1);
    u = zeros(n, 1);
    u(newRowNum) = 1;
    v = (newRow - oldRow)';
    newInverse = inverseA - (inverseA * u * v' * inverseA) ./ (1 + v' * inverseA * u);
end
