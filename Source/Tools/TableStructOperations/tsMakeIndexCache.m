function SCached = tsMakeIndexCache(S, CacheFields)
% ������� ������� ��� �������� ��� ������� �������� ��� ��������� ������
% �����
%Input
% S [struct] - �������� ���������. ��������������, ��� ��������� �����
% ������ ���� "�������":
% S.field1[Nx1 | NxM1]
% S.field2[Nx1 | NxM2]
% ...
% S.fieldk[Nx1 | NxMk]
%
% CacheFields {Cx1 cell} - ������ �����, �� �������� ��������� ������� ���.
% �� ��������� ���������� ��� ����, �� ���������� ����������� � ��������
%
%Output
% SCached [struct] - ���������, ���������� �� �� ����, ��� ��������, �
% ����������� ����� ����� IndexCache ���������� ����:
% IndexCache.<fieldname>Idx.<fieldname> - ����� ���������� �������� � ���� <fieldname>
% IndexCache.<fieldname>Idx.Idx{Nx1 cell} - ��� ��������, N - ����� �������� <fieldname>

Check(tsIsValid(S));

if nargin < 2, CacheFields = fieldnames(S); end

SCached = S;
for f = 1:length(CacheFields)
    fName = CacheFields{f};
    if isstruct(S.(fName)) || iscell(S.(fName)), 
        % TODO: make warning if struct or cell is passed from outside
        continue; 
    end
    fValues = unique(S.(fName));
    eval(['IndexCache.' fName 'Idx.' fName ' = fValues;']);
    fIdx = cell(size(fValues));
    for v = 1:length(fValues)
        fIdx{v} = (S.(fName) == fValues(v));
    end
    eval(['IndexCache.' fName 'Idx.Idx = fIdx;']);
end

SCached.IndexCache = IndexCache;

Check(tsIsValid(SCached));

end