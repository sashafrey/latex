function [sample, sampleClasses] = GenerateSimpleSample(L)

sample = [ mvnrnd( repmat([0 0], L/2, 1), [3 0; 0 3]) ; mvnrnd( repmat([7 7], L/2, 1), [3 0; 0 3]) ];
sampleClasses = [ones(1, L/2) -ones(1, L/2)];

end