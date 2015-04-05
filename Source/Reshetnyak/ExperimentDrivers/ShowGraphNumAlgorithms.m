function ShowGraphNumAlgorithms
    
    L = 200;
    l = L/2;
    eps = 0.05;
    m0 = 0;
    
    %Генерация выборки
    %[sample, sampleClasses] = GenerateSimpleSample(L);
	[sample, sampleClasses] = GenerateCloseClasses(L);
	
    %[sample, sampleClasses] = GenerateStrips(L, m0);
    figure
	hold on
	ind = find(sampleClasses == 1);
    scatter(sample(ind, 1), sample(ind, 2), 30, 'b');
	ind = find(sampleClasses ~= 1);
	scatter(sample(ind, 1), sample(ind, 2), 30, 'r', '*');
    hold off
    
    %Построение всех линейных классификаторов
    R = BuildLinearSet(sample, sampleClasses);
    %PaintAlgorithmsFamily(R);
   
    
    numAlgs = size(R, 1);
   
    familyGraph = BuildFamilyGraph(R);
    
    newErmBound = zeros(1, numAlgs);
	ermBound = zeros(1, numAlgs);
    exactValueEst = zeros(1, numAlgs);
    vBound = zeros(1, numAlgs);
    uniformBound = zeros(1, numAlgs);
  
    numIterations = 1;
    prev = 0;
    for n = 1:numIterations
        n
        
        
% 		ind = [1 randperm(numAlgs - 1) + 1];
%         R(ind, :) = R(1:numAlgs, :);
%         fg = familyGraph;
%         for i = 1:numAlgs
%             familyGraph{ind(i) } = fg{ i}; 
%             for j = 1: size(familyGraph{ind(i) }, 2)
%                 familyGraph{ind(i) }(j) = ind( familyGraph{ ind(i) }(j) );
%             end
%         end
        
    
        %Вычисление разных типов оценок для семейства алгоритмов
             
        exactValueEst = exactValueEst + MonteCarloEstimation(R, l, eps);
        vBound = vBound + VapnikFunctional(R, l, eps);
        %uniformBound = uniformBound + ConnectedFunctional(R, l, eps);
        newErmBound = newErmBound + NewERMFunctional(R, familyGraph, l, eps, 'simple');
		ermBound = ermBound + NewERMFunctional(R, familyGraph, l, eps, 'simple_connect');
    end
    exactValueEst = exactValueEst / numIterations;
    newErmBound = newErmBound / numIterations;
    vBound = vBound / numIterations;
	ermBound = ermBound / numIterations
        %Визуализация
    figure
    hold on
    
    %plot ([1:numAlgs], vBound, 'g');
	plot ([1:numAlgs], ermBound, 'g');
    plot ([1:numAlgs], exactValueEst, 'r');
    plot ([1:numAlgs], newErmBound, 'b');
    %plot ([1:numAlgs], uniformBound, 'k');

    hold off

end