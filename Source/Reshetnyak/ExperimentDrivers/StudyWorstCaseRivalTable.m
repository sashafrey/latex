sampleSize = 800;
trainSize = sampleSize / 2;

eps = 0.05;

shift = [0 : 50];
overfitProb = zeros(1, numel(shift));
profile = ComputeCyclicPolytopeSplittingProfile(sampleSize, 2);
for t = 1 : numel(shift)'
    shift(t)
    [worstProb, mNew] = ComputeWorstCaseRivalTable(shift(t), sampleSize, trainSize, eps(n), 0);
    overfitProb(t) = sum(profile .* worstProb');
end
plot(shift, overfitProb);