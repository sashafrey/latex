%% ��������� �������
load('./esokolov/Experiments/RW/unsep200_new.mat', 'X', 'Y', 'algs_all', 'algsCnt', 'sources', ...
    'sourcesCnt', 'sourcesVects');
L = size(X, 1);
d = size(X, 2);

%% ������� �������� ������� ����������
profileErr_true = zeros(L + 1, 1);
for i = 1:algsCnt
    m = algs_all(i).errCnt;
    profileErr_true(m) = profileErr_true(m) + 1;
end

%% ��������� ������� ���������� � ������� �������������
[profileErr, ~] = estimateScProfile(X, Y, 20000, true);

%% ������� ������ ����� ������ ������ maxLevelForSources �����
maxLevelForSources = 19;
L = size(X, 1);
d = size(X, 2);
[~, algs_src, algsCnt_src, sources_src, sourcesCnt_src] = computeOverfittingBound(X, Y, ...
    L / 2, 0.1, [1 1 1], maxLevelForSources, 1:L, -1);
sourcesVects = convertAlgsFamily(algs_src(sources(1:sourcesCnt)), sourcesCnt, L);

%% �� ������ ������ ������ ��������� ���������
maxLevel = 35;

%% ��������� ����� ������ � ������ maxLevel �����
Vt = getVt(X, Y, 30000, maxLevel);

%% �� ������ ������
algs_all = algs_all(1:algsCnt);

%%
save('./esokolov/Experiments/RW/unsep200_rw_ready.mat', ...
    'X', 'Y', 'algs_all', 'algsCnt', 'sources', 'sourcesCnt', 'sourcesVects', ...
    'profileErr_true', 'profileErr', 'algs_src', 'algsCnt_src', 'sources_src', 'sourcesCnt_src', ...
    'maxLevel', 'Vt');