function [sample, sampleClasses] = GenerateDiffClasses(L)
	sample = [ mvnrnd( repmat([0 0], L, 1), [1 0; 0 1])];
	sampleClasses = round(rand(L, 1));
end