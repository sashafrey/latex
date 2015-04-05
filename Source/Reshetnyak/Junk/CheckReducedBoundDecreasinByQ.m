
sampleSize = 100;
trainSize = 40;
eps = 0.05;
shift = 3;
maxQ = 20;

[worstProb, mNew, qNew] = ComputeWorstCaseRivalTable(shift, sampleSize, trainSize, eps, maxQ);

for m = 0 : sampleSize
    for q = 0 : maxQ
        t = mNew(m + 1, q + 1);
        if (t ~= 0) && (qNew(m + 1, q + 1) ~= max(0, q - (t - m + shift) / 2))
            m
            q
        end
    end
end
m

