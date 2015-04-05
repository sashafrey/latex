function StudyWorstCaseForLinearSeparableSample

% 	sampleSize  = 1000;
% 	trainSize = floor(sampleSize / 2);
% 	eps = 0.05;
% 	
% 	overfitProb = zeros(1, trainSize);
% 	for numZerosInRow = 1:trainSize
% 		overfitProb(numZerosInRow) = ComputeFractionOf0_1SequencesWithFixedCountOfZerosInARow(sampleSize,trainSize, numZerosInRow);
% 	end

    dim = 3
eps = 0.05
 	samplePoints = [2:2:800]; 
 	overfitProb = zeros(1, numel(samplePoints) );
 	for i = 1:numel(samplePoints)
 		trainSize = floor(samplePoints(i) / 2);
 		overfitProb(i) =  ComputeFractionOf0_1SequencesWithFixedCountOfZerosInARow(...
 			samplePoints(i),trainSize, ceil(trainSize * eps) );
 	end
	%CreateFigure1(samplePoints, overfitProb);
    plot(samplePoints, overfitProb);
    ylabel('Вероятность переобучения');
    xlabel('Длина полной выборки');
	overfitProb
end