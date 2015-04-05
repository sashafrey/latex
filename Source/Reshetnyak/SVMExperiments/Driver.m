sampleSize = 200;
eps = 0.05;
l = sampleSize / 2;

[sample, answers] = GenerateSin1DimSample(sampleSize, 10);

numTries = 1000;

overfitProb = 0;
objectsErrorFreq = zeros(sampleSize, 1);
for t = 1:numTries
    t
    ind = randperm(sampleSize);
    svmStruct = svmtrain( sample(ind(1:l), :), answers(ind(1 : l), :), 'Kernel_Function', 'polynomial', 'Polyorder', 2);
    svmAnswers = svmclassify(svmStruct, sample);
    error = (svmAnswers ~= answers);
    objectsErrorFreq = objectsErrorFreq + error;
    trainError = sum(error( ind(1 : l) ) );
    testError = sum(error( ind(l + 1 : sampleSize) ) );
    overfitProb = overfitProb + (testError/(sampleSize - l) - trainError/l >= eps);
end

objectsErrorFreq = objectsErrorFreq / numTries;
hold on
stem(sample(answers == 1, :), objectsErrorFreq(answers == 1), 'fill');
stem_handle = stem(sample(answers == -1, :), objectsErrorFreq(answers == -1), 'fill');
set(stem_handle,'MarkerFaceColor','red')
set(stem_handle,'MarkerFaceColor','red')
set( get(stem_handle,'BaseLine'),'LineStyle','--','Color','red');


overfitProb = overfitProb / numTries


