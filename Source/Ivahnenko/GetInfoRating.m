function rating=GetInfoRating(p)
pCount = size(p.perm,1);
rating = zeros(pCount,1);
Pos = sum(p.sample(:,end));
Neg = p.L-Pos;
for i=1:pCount
    sample = p.sample(:,[p.perm(i,:) end]);
    [coord ~] = GetCoords(sample(:,1:end-1));
    maxInfo = -1;
    bestAlg = [];
    bestAC = -1;
    for alg=coord'
        [rC aC] = Check(1:p.L, alg);
        info = CalcInfo(p, Pos, rC, Neg, aC-rC);
        if (info > maxInfo)
            maxInfo = info;
            bestAlg = alg;
            bestAC = aC;
        end;
    end;
    rating(i) = maxInfo;
end

function [rightCovered, allCovered] = Check(set,a)
    a_f = repmat(a',length(set),1);
    delta = sample(set,1:end-1)-a_f;
    cover = all(delta<=0, 2);
    rightCovered = sum(cover &(sample(set,end)==1));
    allCovered = sum(cover);
end

end