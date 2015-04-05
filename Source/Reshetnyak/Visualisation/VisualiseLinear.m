%Генерирует выборку, по выборке строит семейство линейных классификаторов
%и рисует его граф
%
function VisualiseLinear()
    
    L = 6 %Длина выборки
    %Генерация выборки
    [sample, sampleClasses] = GenerateCloseClasses(L);
	%[sample, sampleClasses] = GenerateSimpleSample(L);
    %Визуализация выборки
	hold on
	ind = find(sampleClasses == 1);
    scatter(sample(ind, 1), sample(ind, 2), 50, 'b', 'filled');
	ind = find(sampleClasses ~= 1);
	scatter(sample(ind, 1), sample(ind, 2), 50, 'r');
    hold off
    %Построение семейства линейных классификаторов
    %algs = VisualizeConjunctionSet(sample, sampleClasses);
    algs = BuildLinearSet(sample, sampleClasses);
    graph = BuildFamilyGraph(algs);
    numEdges = sum(cellfun('length', graph))
    
	%algs = BuildkNNSet(sample, sampleClasses, 8, 3);
    %VisualizeLinearSet(sample, sampleClasses);
    %Визуализация графа алгоритмов
    PaintAlgorithmsFamily(algs);
    SaveAdjacencyMatrix('graph.txt', algs);
    
end