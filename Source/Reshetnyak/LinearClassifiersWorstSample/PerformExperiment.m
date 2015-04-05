sampleSize = 400
trainSize = sampleSize / 2
results = zeros(1, trainSize);
for m = 1:trainSize
    results(m) = ComputeFractionOf0_1SequencesWithFixedCountOfZerosInARow(sampleSize, trainSize, m);
end


plot(results);
xlabel('Число ошибок');
ylabel('Вероятность переобучения');
