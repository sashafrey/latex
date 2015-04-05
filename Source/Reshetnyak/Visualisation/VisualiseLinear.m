%���������� �������, �� ������� ������ ��������� �������� ���������������
%� ������ ��� ����
%
function VisualiseLinear()
    
    L = 6 %����� �������
    %��������� �������
    [sample, sampleClasses] = GenerateCloseClasses(L);
	%[sample, sampleClasses] = GenerateSimpleSample(L);
    %������������ �������
	hold on
	ind = find(sampleClasses == 1);
    scatter(sample(ind, 1), sample(ind, 2), 50, 'b', 'filled');
	ind = find(sampleClasses ~= 1);
	scatter(sample(ind, 1), sample(ind, 2), 50, 'r');
    hold off
    %���������� ��������� �������� ���������������
    %algs = VisualizeConjunctionSet(sample, sampleClasses);
    algs = BuildLinearSet(sample, sampleClasses);
    graph = BuildFamilyGraph(algs);
    numEdges = sum(cellfun('length', graph))
    
	%algs = BuildkNNSet(sample, sampleClasses, 8, 3);
    %VisualizeLinearSet(sample, sampleClasses);
    %������������ ����� ����������
    PaintAlgorithmsFamily(algs);
    SaveAdjacencyMatrix('graph.txt', algs);
    
end