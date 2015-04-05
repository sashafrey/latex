%вычисляем фиксированные точки для предикатов на основе полного перебора
%реально может работать для выборок не больше 20 объектов.
%Записывает результат в p.pa
%p.pa{:,1} - фиксированные точки на обучении
%p.pa{:,2} - фиксированные точки на контроле
%p.pa{:,3} - разбиения на которых был выбран соответсвующий предикат

function p = GetFixPointsBF(p)
%взяли все разбиения. будем перебирать все возможные обучения.
perm= nchoosek(1:p.L, p.L/2);
algCount = size(p.coord,1);
pa = cell(algCount,3); %число фиксированных точек в алгоритме на обучении и контроле
for i=1:algCount
    pa{i,1} = 1:p.L;
    pa{i,2} = 1:p.L;
    pa{i,3} = [];
end;

distCount = size(perm,1);
for l=1:distCount
    %взяли обучение, контроль
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