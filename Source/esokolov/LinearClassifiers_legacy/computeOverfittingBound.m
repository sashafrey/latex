function [bound algs algsCnt sources sourcesCnt] = computeOverfittingBound(X, Y, l, eps, a0, maxLevel, goodObjects, maxSteps, varargin)
    display = true;
    if (length(varargin) == 1 && varargin{1} == false)
        display = false;
    end
    % a0 - нормаль к какой-нибудь гиперплоскости; чем ближе она к
    % оптимальной - тем лучше
    L = size(X, 1);
    d = size(X, 2);
    
    binCoef = zeros(L + 1, L + 1);
    for i = 0:L
        for j = 0:L
            binCoef(i + 1, j + 1) = cnk(i, j);
        end
    end
    
    X = [X ones(L, 1)]; % фиктивный признак - чтобы у нас были линейные классификаторы с порогом
    a0 = a0 ./ abs(a0(d + 1));  % хотим, чтобы порог был +1 или -1
    
    % поиск вершины €чейки, содержащей a0
    currDescr = zeros(1, d);    % на пересечении каких гиперплоскостей лежит currVertex?
    currVertex = a0;
    A = eye(d + 1);
    b = a0';
    for i = 1:d
        b(i) = 0;
        % ищем гиперплоскость, наход€щуюс€ ближе всего к текущей точке
        bestDist = +Inf;
        bestIdx = 0;
        for j = 1:L
            if (ismember(j, currDescr))
                continue;
            end
            A(i, :) = X(j, :);
            p = A \ b;
            if (dist(p', currVertex) < bestDist)
                bestDist = dist(p', currVertex);
                bestIdx = j;
            end
        end
        currDescr(i) = bestIdx;
        A(i, :) = X(bestIdx, :);
        currVertex = (A \ b)';
        b((i + 1):d) = currVertex((i + 1):d);
    end
    currErrVect = zeros(1, L);
    for i = 1:L
        currErrVect(i) = (sign(X(i, :) * currVertex') ~= Y(i));
    end
    currErrVect(currDescr) = 0;
    
    if (L - sum(currErrVect) - d < sum(currErrVect))
        currErrVect = ~currErrVect;
        currErrVect(currDescr) = 0;
        currVertex = -currVertex;
    end
    
    % структуры дл€ хранени€ найденных алгоритмов
    maxAlgCnt = 10000;
    algs(maxAlgCnt) = struct('vertex', [], 'errVect', [], 'errCnt', [], ...
        'descr', [], 'lowerNeighsCnt', [], 'upperNeighsCnt', [], ...
        'lowerNeighbours', [], 'upperNeighbours', [], 'bound', []);
    algs(1) = struct('vertex', currVertex, 'errVect', currErrVect, 'errCnt', sum(currErrVect), ...
        'descr', sort(currDescr), 'lowerNeighsCnt', 0, 'upperNeighsCnt', 0, ...
        'lowerNeighbours', [], 'upperNeighbours', [], 'bound', 0);
    algsCnt = 1;
    
    % истоки SC-графа
    sources = zeros(maxAlgCnt, 1);
    sourcesCnt = 0;
    
    % очереди
    maxQueueSize = 10000;
    queue1 = struct('q', zeros(L + 1, maxQueueSize), 'start', ones(L + 1, 1), 'size', zeros(L + 1, 1)); % алгоритмы, дл€ которых нужно найти соседей
    queue2 = struct('q', zeros(L + 1, maxQueueSize), 'start', ones(L + 1, 1), 'size', zeros(L + 1, 1)); % алгоритмы, дл€ которых нужно вычислить оценку
    
    queue1.q(algs(1).errCnt + 1, 1) = 1;
    queue1.size(algs(1).errCnt + 1) = 1;
    
    bound = 0;
    %profile = zeros([L+1 L+1 L+1]);
    
    stepsCnt = 0;
    
    while true
        % ищем непустые очереди с минимальными номерами
        queueNum1 = Inf;
        queueNum2 = Inf;
        for i = 1:(L + 1)
            if (isinf(queueNum1) && queue1.size(i) > 0)
                queueNum1 = i;
            end
            if (isinf(queueNum2) && queue2.size(i) > 0)
                queueNum2 = i;
            end
            if (~isinf(queueNum1) && ~isinf(queueNum2))
                break;
            end
        end
        
        if (isinf(min(queueNum1, queueNum2)))
            break;  % очереди пустые
        end
        
        if (queueNum1 < queueNum2)
            [queue1, currAlgNum] = popFromQueue(queue1, queueNum1 - 1);
            [algs, algsCnt, queue1, queue2, sources, sourcesCnt] = ...
                search1(queue1, queue2, algs, algsCnt, currAlgNum, sources, sourcesCnt, X, Y, goodObjects);
            if (display)
                fprintf('Level: %d\n', algs(currAlgNum).errCnt);
            end
        else
            [queue2, currAlgNum] = popFromQueue(queue2, queueNum2 - 1);
            
            if (sum(algs(currAlgNum).errCnt > maxLevel))
                return;
            end
            
            [b, queue1] = search2(queue1, algs, algsCnt, currAlgNum, sources, sourcesCnt, X, l, eps, binCoef);
            algs(currAlgNum).bound = b;
            bound = bound + b;
            if (display)
                fprintf('%f\n', bound);
            end
        end
        
        stepsCnt = stepsCnt + 1;
        if (maxSteps > 0 && stepsCnt >= maxSteps)
            break;
        end
    end
end

function d = dist(x, y)
    d = sum((x - y) .^ 2);
end

function [newAlgs, newAlgsCnt, newQueue1, newQueue2, newSources, newSourcesCnt] = ...
    search1(queue1, queue2, algs, algsCnt, currAlgNum, sources, sourcesCnt, X, Y, goodObjects)
    
    newAlgs = algs;
    newAlgsCnt = algsCnt;
    newQueue1 = queue1;
    newQueue2 = queue2;
    newSources = sources;
    newSourcesCnt = sourcesCnt;
    
    if (algs(currAlgNum).lowerNeighsCnt ~= 0 || algs(currAlgNum).upperNeighsCnt ~= 0)
        return; % уже были в этой вершине
    end    
    
    [newAlgs, newAlgsCnt, newSources, newSourcesCnt] = getNeighbours(algs, ...
        algsCnt, currAlgNum, sources, sourcesCnt, X, Y, goodObjects);
    newQueue2 = addToQueue(newQueue2, newAlgs(currAlgNum).errCnt, currAlgNum); % запоминаем, что нужно сделать вторую часть обработки
    % но сначала нужно побывать во всех алгоритмах, которые наход€тс€ ниже нашего
    for i = 1:newAlgs(currAlgNum).lowerNeighsCnt
        num = newAlgs(currAlgNum).lowerNeighbours(i);
        if (newAlgs(num).lowerNeighsCnt == 0 && newAlgs(num).upperNeighsCnt == 0)   % если мы здесь еще не были
            newQueue1 = addToQueue(newQueue1, newAlgs(num).errCnt, num);
        end
    end    
end

function [bound, newQueue1] = search2(queue1, algs, algsCnt, currAlgNum, sources, sourcesCnt, X, l, eps, binCoef)
    L = size(X, 1);
    d = size(X, 2) - 1;
    
    newQueue1 = queue1;
    
    % считаем вклад алгоритма в оценку
    %rVect = zeros(1, L);
    %for i = 1:sourcesCnt
    %    diffErr = algs(currAlgNum).errVect - algs(sources(i)).errVect;
    %    if (diffErr >= 0)
    %        rVect((diffErr) == 1) = 1;
    %    end
    %end
    %r = sum(rVect); % неоптимальность алгоритма
    %q = algs(currAlgNum).upperNeighsCnt; % св€зность
    %m = algs(currAlgNum).errCnt;
    %bound = (cnk(L - q - r, l - q) / cnk(L, l)) * ...
    %    hhDistr(L - q - r, l - q, m - r, floor((l/L) * (m - eps * (L - l))));
    m = algs(currAlgNum).errCnt;
    u = algs(currAlgNum).upperNeighsCnt;
    bound = Inf;
    for j = 1:sourcesCnt
        a = sum(algs(currAlgNum).errVect < algs(sources(j)).errVect);
        b = sum(algs(currAlgNum).errVect > algs(sources(j)).errVect);
        currBound = 0;
        for t = 0:min(a, b)
            currBound = currBound + ((my_cnk(b, t, binCoef) * ...
                my_cnk(L - u - b, l - u - t, binCoef)) / my_cnk(L, l, binCoef)) * ...
                my_hhDistr(L - u - b, l - u - t, m - b, floor((l/L) * (m - eps * (L - l))) - t, binCoef);
        end
        bound = min(bound, currBound);
    end
    
    % если вклад маленький, то хватит уже
    if (bound < 0)
        return;
    end
    
    % добавл€ем в очередь алгоритмы из верхней полуокрестности
    for i = 1:algs(currAlgNum).upperNeighsCnt
        num = algs(currAlgNum).upperNeighbours(i);
        if (algs(num).lowerNeighsCnt == 0 && algs(num).upperNeighsCnt == 0)   % если мы здесь еще не были
            newQueue1 = addToQueue(newQueue1, algs(num).errCnt, num);
        end
    end
end

function [newQueue] = addToQueue(queue, level, element)
    queue.size(level + 1) = queue.size(level + 1) + 1;
    pos = queue.start(level + 1) + queue.size(level + 1) - 1;
    if (pos <= size(queue.q, 2))
        queue.q(level + 1, pos) = element;
    else
        pos = mod(pos, size(queue.q, 2)) + 1;
        queue.q(level + 1, pos) = element;
    end
    if (pos == queue.start(level + 1) && queue.size(level + 1) > 1)
        fprintf('Size of queue is too small\n');
        pause;
    end
    newQueue = queue;
end

function [newQueue, element] = popFromQueue(queue, level)
    pos = queue.start(level + 1);
	element = queue.q(level + 1, pos);
    queue.start(level + 1) = queue.start(level + 1) + 1;
    if (queue.start(level + 1) > size(queue.q, 2))
        queue.start(level + 1) = mod(queue.start(level + 1), size(queue.q, 2)) + 1;
    end
    queue.size(level + 1) = queue.size(level + 1) - 1;
    newQueue = queue;
end

function res = my_cnk(n, k, binCoef)
    if (n >= 0 && k >= 0 && k <= n)
        res = binCoef(n + 1, k + 1);
    else
        res = 0;
    end
end

function h = my_hhDistr(L, l, m, s, binCoef)
    h = 0;
    if (L >= 0 && l >= 0 && l <= L)
        for i = 0:s
            h = h + my_cnk(m, i, binCoef) * my_cnk(L - m, l - i, binCoef) / my_cnk(L, l, binCoef);
        end
    end
end