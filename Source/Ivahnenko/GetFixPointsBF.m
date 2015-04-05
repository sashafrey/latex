%��������� ������������� ����� ��� ���������� �� ������ ������� ��������
%������� ����� �������� ��� ������� �� ������ 20 ��������.
%���������� ��������� � p.pa
%p.pa{:,1} - ������������� ����� �� ��������
%p.pa{:,2} - ������������� ����� �� ��������
%p.pa{:,3} - ��������� �� ������� ��� ������ �������������� ��������

function p = GetFixPointsBF(p)
%����� ��� ���������. ����� ���������� ��� ��������� ��������.
perm= nchoosek(1:p.L, p.L/2);
algCount = size(p.coord,1);
pa = cell(algCount,3); %����� ������������� ����� � ��������� �� �������� � ��������
for i=1:algCount
    pa{i,1} = 1:p.L;
    pa{i,2} = 1:p.L;
    pa{i,3} = [];
end;

distCount = size(perm,1);
for l=1:distCount
    %����� ��������, ��������
    minErrTrain = p.L;
    maxErrTest = 0;
    appAlg = [];
    for i=algCount:-1:1
        a = p.coord(i,:)';
        errTrain = CalcErrCount(l,a);
        errTest = p.errorsCount(i)-errTrain;
        if (errTrain<minErrTrain || (errTrain==minErrTrain && errTest>maxErrTest))
            appAlg = i;
            minErrTrain=errTrain;
            maxErrTest = errTest;
        end;
    end;
    pa{appAlg,1} = intersect(pa{appAlg,1},perm(l,:));
    pa{appAlg,2} = setdiff(pa{appAlg,2},perm(l,:));
    pa{appAlg,3}(end+1) = l;
end;
p.pa = pa;

function errCount = CalcErrCount(razb, alg)
    samp = p.sample(perm(razb,:),1:end-1);
    b = repmat(alg',p.L/2,1);
    
    errCount = sum(all(samp-b<=0, 2)~=(p.sample(perm(razb,:),end)==1));
end
end