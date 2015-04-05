function rating=GetQRating(p, N)
if (max(p.perm(:))>p.n)
    disp('Ошибка в указании признаков для тестирования.');
    disp('Номер признака больше числа столбцов данной выборки.');
    return;
end;

r = p.rand;
l = p.L/2;
pCount = size(p.perm,1);
Qeps = zeros(2*l+1,pCount);
rating = zeros(pCount,1);

for rp=1:pCount
    sample = p.sample(:,[p.perm(rp,:) end]);
    for i=1:N
        ix = r.randperm(p.L);
        trainSet = sort(ix(1:l));
        testSet = sort(ix(l+1:end));
        coord = GetCoords(sample(trainSet,1:end-1),p.L);
        minTrainErr = p.L;
        maxTestErr = 0;
        bestAlg = [];
        bestAC = -1;
        for alg=coord'
            trainErr = Check(trainSet, alg);
            if (trainErr<minTrainErr)
                minTrainErr = trainErr;
                [aC rC] = Check(testSet, alg);
                maxTestErr = aC-rC;
                bestAC = aC;
                bestAlg = alg;
            elseif (trainErr==minTrainErr)
                [aC rC] = Check(testSet, alg);
                if (aC-rC>maxTestErr)
                    maxTestErr = aC-rC;
                    bestAC = aC;
                    bestAlg = alg;
                end;
            end;
        end;
        rating(rp) = rating(rp)+maxTestErr;
        %Qeps(maxTestErr+l+1,rp) = Qeps(maxTestErr+l+1,rp)+1;
    end;
end;

rating = rating/N;
%Qeps = Qeps./N;
%for i=2:2*l+1
%    Qeps(i,:)=Qeps(i,:)+Qeps(i-1,:);
%    rating((rating==0)' &(Qeps(i,:)>0.9)) = i;
%end;
%Qeps = 1-Qeps;
%rating = find(Qeps<0.1, 1, 'first');

function [allCovered rightCovered] = Check(set,a)
    a_f = repmat(a',length(set),1);
    delta = sample(set,1:end-1)-a_f;
    cover = all(delta<=0, 2);
    %errs = sum(cover ==sample(set,end));
    rightCovered = sum(cover &(sample(set,end)==1));
    allCovered = sum(cover);
    %errs = allCovered-rightCovered;
    %allCovered = sum(cover);
    %errs = allCovered-rightCovered;
end

end