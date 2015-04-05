%% Функция вычисления точного значения Qeps методом Монте-Карло
% Вычисление происходит по определению, как доля разбиений на которых
% переобученность (разница в долях ошибок на контроле и обучении) больше
% eps. Рекомендуется подавать на вход данные с четным числом объектов,
% потому что l = L/2. Тогда eps может принимать дискретные значения с шагом
% 1/2L
function [Qeps] = CalcQ(p,N)
r = RandStream('mt19937ar','Seed',1);
l = p.L/2;
Qeps = zeros(2*l+1,2);
Qeps(:,1) = -1:1/l:1;
sample = p.sample;
for i=1:N
    ix = r.randperm(p.L);
    trainSet = ix(1:l);
    
    bestErrTrain = l;
    bestErrTest = 0;
    %считаем переобученность
    for aIdx = length(p.errors):-1:1
        [aC rC] = Check(trainSet, p.coord(aIdx,:));
        errTrain = aC-rC;
        errTest = p.errorsCount(aIdx)-errTrain;
        if (errTrain<bestErrTrain || (errTrain==bestErrTrain && errTest>bestErrTest))
            bestErrTrain = errTrain;
            bestErrTest = errTest;
        end;
    end;
    
    delta = bestErrTest-bestErrTrain;
    Qeps(delta+l+1,2) = Qeps(delta+l+1,2)+1;
end;

i = find(Qeps(:,2)>0);
Qeps(i,2) = Qeps(i,2)/N;


function [allCovered rightCovered] = Check(set,a)
    allCovered = length(set);
    a_f = repmat(a,allCovered,1);
    delta = sample(set,1:end-1)-a_f;
    cover = all(delta<=0, 2);
    rightCovered = sum(cover ==(sample(set,end)==1));
end

end