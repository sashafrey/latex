function q = CalcOneStep(p, trainSet, testSet)
trainLen = length(trainSet);
testLen = length(testSet);
trainPos = sum(p.sample(trainSet,end));
trainNeg = trainLen-trainPos;
coord = GetCoords(p.sample(trainSet,1:end-1),p.L);

q.bestInfo = -1;
q.bestDeltaBoth = 0;
q.bestDeltaPos=0;
q.bestDeltaNeg=0;
%считаем переобученность
for alg = coord'
    [aC rC] = Check(trainSet, alg');
    infoTrain = CalcInfo(p, trainPos, rC+p.pAdd, trainNeg, aC-rC+p.nAdd);

    if (infoTrain>q.bestInfo)
        q.bestAlg = alg';
        q.bestInfo = infoTrain;
        [aCTest rCTest] = Check(testSet, alg');
        q.bestDeltaBoth = (aCTest-rCTest+testLen-rCTest)-(aC-rC+trainLen-rC);
        q.bestDeltaPos = rCTest-rC;
        q.bestDeltaNeg = (aCTest-rCTest)-(aC-rC);
        q.trainPos = rC;
        q.trainNeg = aC-rC;
        q.testPos = rCTest;
        q.testNeg = aCTest-rCTest;
    end;
end;

function [allCovered rightCovered] = Check(set,a)
    a_f = repmat(a,length(set),1);
    delta = p.sample(set,1:end-1)-a_f;
    cover = all(delta<=0, 2);
    rightCovered = sum(cover & cover ==(p.sample(set,end)==1));
    allCovered = sum(cover);
end

end