function alg = initLinearAlgSimpleStructure(w, errVect, errCnt, ...
    lowerNeighsCnt, upperNeighsCnt, lowerNeighbours, upperNeighbours)
% ���������� ���������, ����������� �������� ������������

    alg = struct('w', [], ...   % ����, ��������������� ������� ��������������
                 'errVect', [], ... % ������ ������
                 'errCnt', [], ...  % ����� ������
                 'lowerNeighsCnt', [], ... % ����� ������ �������
                 'upperNeighsCnt', [], ...  % ����� ������� �������
                 'lowerNeighbours', [], ... % ������� ������ �������
                 ...                        % (���� ��������� �������� � ����� �������)
                 'upperNeighbours', []... % ������� ������� �������
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
