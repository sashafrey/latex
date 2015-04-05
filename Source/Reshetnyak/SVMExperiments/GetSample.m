function [sample, answers] = GenerateSin1DimSample(sampleSize, baseLength)
    sample = [0: baseLength/sampleSize baseLength - sampleSize]';
    answers = sign( sin(sample) );
    answers(answers == 0) = 1;
end