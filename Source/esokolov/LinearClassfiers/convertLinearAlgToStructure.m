function alg = convertLinearAlgToStructure(w, X, Y)
% ��� ��������� ��������� �������������� � ������ w ��������� ���������,
% ��������� � initLinearAlgStructure().
% ������� �������� ������������� w ������������� � ������������� � ����� ��
% �������� ������, �� ���������� ����� d ����� �������.
%
% ��������������, ��� ��������� ������� � X --- �����������

    % w ������ ���� ��������
    if size(w, 1) == 1
        w = w';
    end

    L = size(X, 1);
    d = size(X, 2) - 1; % ����������� ������� �� ������� �� �������

    w = w ./ abs(w(d + 1)); % ��� ��� ��������� ������ ���� ������ ������� �� ������

    currDescr = zeros(1, d);  % �� ����������� ����� ��������������� ����� currVertex?
    currVertex = w;
    A = eye(d + 1);
    b = w;
    
    % ������� �������������� � d ��������
    for i = 1:d
        b(i) = 0;
        % ���� ��������������, ����������� ����� ����� � ������� �����
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
    
    % ������� ������ ������
    currErrVect = (sign(X * currVertex) ~= Y);
    
    errVectInit = (sign(X * w) ~= Y);
    currErrVect(currDescr) = errVectInit(currDescr);
    
    if (sum(currErrVect) < sum(errVectInit))
        currErrVect = 1 - currErrVect;
        currErrVect(currDescr) = errVectInit(currDescr);
        currVertex = -currVertex;
    end
    
    % ��������� ���������
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
