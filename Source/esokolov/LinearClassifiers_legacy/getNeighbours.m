function [newAlgs, newAlgsCnt, newSources, newSourcesCnt] = getNeighbours(algs, ...
        algsCnt, currAlgNum, sources, sourcesCnt, X, Y, goodObjects)
    
    L = size(X, 1);
    d = size(X, 2) - 1;
    
    alg = algs(currAlgNum);
    lowerNeighbours = zeros(1, L);
    upperNeighbours = zeros(1, L);
    lowerNeighsCnt = 0;
    upperNeighsCnt = 0;
    
    % достаточно ли L?
    descrs = zeros(L, d);   % описания всех вершин нашей ячейки
    vertices = zeros(L, d + 1);
    descrs(1, :) = sort(alg.descr);
    vertices(1, :) = alg.vertex;
    descrCnt = 1;
    
    i = 1;
    while (i <= descrCnt)
        [descrs, vertices, descrCnt] = getIncidentVertices(descrs, vertices, descrCnt, i, alg, X, Y, goodObjects);
        i = i + 1;
    end
    
    for i = 1:descrCnt
        newErrVect = alg.errVect;
        for k = 1:d
            newErrVect(descrs(i, k)) = ~newErrVect(descrs(i, k));
            newErrCnt = alg.errCnt;
            if (newErrVect(descrs(i, k)) == 0)
                newErrCnt = newErrCnt - 1;
            else
                newErrCnt = newErrCnt + 1;
            end

            % findAlgorithm
            idx = -1;
            for z = 1:algsCnt
                if (algs(z).errVect == newErrVect)
                    idx = z;
                    break;
                end
            end

            % insertAlgorithm
            if (idx == -1)
                algsCnt = algsCnt + 1;
                algs(algsCnt) = struct('vertex', vertices(i, :), 'errVect', newErrVect, 'errCnt', newErrCnt, ...
                   'descr', descrs(i, :), 'lowerNeighsCnt', 0, 'upperNeighsCnt', 0, ...
                   'lowerNeighbours', [], 'upperNeighbours', [], 'bound', []);
                idx = algsCnt;
            end

            if (newErrVect(descrs(i, k)) == 0)
                lowerNeighsCnt = lowerNeighsCnt + 1;
                lowerNeighbours(lowerNeighsCnt) = idx;
            else
                upperNeighsCnt = upperNeighsCnt + 1;
                upperNeighbours(upperNeighsCnt) = idx;
            end
            newErrVect(descrs(i, k)) = ~newErrVect(descrs(i, k));
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
    algs(currAlgNum) = alg;
    
    if (alg.lowerNeighsCnt == 0)
        sourcesCnt = sourcesCnt + 1;
        sources(sourcesCnt) = currAlgNum;
    end
    newAlgs = algs;
    newAlgsCnt = algsCnt;
    newSources = sources;
    newSourcesCnt = sourcesCnt;
end

function [descrs, vertices, descrCnt] = getIncidentVertices(oldDescrs, oldVertices, oldDescrCnt, currDescrNum, alg, X, Y, goodObjects)
    L = size(X, 1);
    d = size(X, 2) - 1;
    
    descrs = oldDescrs;
    descrCnt = oldDescrCnt;
    vertices = oldVertices;
    
    currDescr = descrs(currDescrNum, :);

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
            %for j = setdiff(1:L, currDescr)
            for j = setdiff(goodObjects, currDescr)
                %A(i, :) = X(j, :);
                %newVertex = (A \ b)';
                newVertex = recalculateInverse(inverseA, i, X(j, :), A(i, :)) * b;
                newVertex = newVertex';
                if (closestPoint(1) == -1)
                    closestPoint(1) = j;
                    minDist(1) = dist(vertices(currDescrNum, :), newVertex);
                    leftVect = newVertex - vertices(currDescrNum, :);
                else
                    if ((newVertex - vertices(currDescrNum, :)) * leftVect' > 0)  % если эта точка по левую сторону
                        idx = 1;
                    else
                        idx = 2;
                    end
                    newDist = dist(vertices(currDescrNum, :), newVertex);
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
                newErrVect = (sign(X * newVertex) ~= Y)';
                newErrVect(newDescr) = alg.errVect(newDescr);
                if (newErrVect == alg.errVect)  % нашли еще одну вершину ячейки
                    isNewDescr = 1;
                    sortedNewDescr = sort(newDescr);
                    for k = 1:descrCnt
                        if (sortedNewDescr == descrs(k, :))
                            isNewDescr = 0;
                            break;
                        end
                    end
                    if isNewDescr
                        descrCnt = descrCnt + 1;
                        descrs(descrCnt, :) = sortedNewDescr;
                        vertices(descrCnt, :) = newVertex;
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
