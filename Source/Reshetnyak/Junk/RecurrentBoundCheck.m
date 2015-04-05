function RecurrentBoundCheck()

    L = 32;
    l = 16;
    eps = 0.3;
    
    [sample, sampleClasses] = GenerateSimpleSample(L);
    
    algs = BuildLinearSet(sample, sampleClasses);
	PaintAlgorithmsFamily(algs);
	
    
    recBound = RecurrentBoundCalculation(algs, l, eps);
    exactValue = ExactFunctional(algs, l, eps);
    
	
    figure
	hold on
    plot(1:size(algs, 1), recBound, 'r');
    plot(1:size(algs, 1), exactValue, 'g');
	plot(1:size(algs, 1), exactValue - recBound, 'b');
	hold off
	 
end