%Функция строит график зависимости оценок от длины выборки
function ShowLGraph
   
    eps = 0.05;

    Lind =  [100:4:400];

    vapnikBound = zeros( size(Lind) );
    exactValueEst  = zeros( size(Lind) );
    ermBound = zeros( size(Lind) );
    %ermBound2 = zeros( size(Lind) );
    
    numIterations = 1;
    
    R = [];
    'Iteration number'
 
    for n =1:numel(Lind)
		sampleSize = Lind(n);
        sampleSize
        trainSize = sampleSize/2;
		
		%Генерация выборки
        [sample, sampleClasses] = GenerateSimpleSample(sampleSize);
        %[sample, sampleClasses] = GenerateRandomSample(L, 2);
       
        %Построение семейства алгоритмов по выборке
        %R = PruneAlgorithmSet( BuildLinearSet(sample, sampleClasses), l );
		algs = BuildLinearSet(sample, sampleClasses);
        exactValueEst(n) = MonteCarloEstimation(algs, trainSize, eps, 20000);
        ermBound(n) = ERMFunctional(algs, BuildFamilyGraph(algs), trainSize, eps, 'simple');
        vapnikBound(n) = VapnikFunctional(algs, trainSize, eps);
		
	end
   
    figure
    hold on

    plot ([Lind], vapnikBound, 'g');
    plot ([Lind], exactValueEst, 'r');
    plot ([Lind], ermBound, 'b');
    %plot ([Lind], ermBound2, 'y');
    hold off

    save LData

end