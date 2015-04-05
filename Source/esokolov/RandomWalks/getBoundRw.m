function bound = getBoundRw(X, Y, w, ell, eps, boundType)
% ���������� ������ ����������� ������������ �� ����������� ��������� w.
% ����������: � ������� X ���� ����������� �������, � ��� �������������
% ��������� ������� � X

    if ~exist('boundType', 'var')
        boundType = 'SC_sources';
    end
    
    L = size(X, 1);
    d = size(X, 2) - 1;

    % ��������� �������� ������������� <w, x> � ������� ��� ���
    % �������������
%     alg_start = convertLinearAlgToStructure(w, X, Y);
    alg_start = convertLinearAlgToSimpleStructure(w, X, Y);
    
    % ��������� ��������� ��������� �� alg_start
    iterCnt_rw = 500;
    maxLevel_rw = alg_start.errCnt + 15;
%     [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
%         alg_start, iterCnt_rw, maxLevel_rw, @getNeighboursLC_rw, true);
    [algs_rw, corrections_rw] = random_walk_simple(X, Y, ...
        alg_start, iterCnt_rw, maxLevel_rw, ...
        @(alg, X, Y) getNeighboursLC_rays(alg, X, Y, 50), ...
        true);
    
    % ��� ���������� ���������� ��� ����������� ������ �� V_t --- �����
    % ������ � ����� � 1-�� �� maxLevel_rw
    iterCnt_Vt = 10000;
    maxLevel_Vt = maxLevel_rw;
    %Vt = getVt(X, Y, iterCnt_Vt, maxLevel_Vt, false);    
    % ��������!!!
    Vt = 1;
    
    % �������� ������� �� ������ ����� �� �����, ������� ������ �� ������
    % ��������� ��������� �� algs_rw
    sources = findSourcesInSample(algs_rw);
    
    % ��������� ������ �� ������� ����������, ���������� � ����������
    % ���������� ���������
    bound = getSrwBoundEstimate(algs_rw, corrections_rw, sources, L, ell, ...
        eps, Vt, boundType);
end
