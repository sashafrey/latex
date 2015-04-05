L = 100;
eps = 0.1;

Lind =  [100:4:100];

cnt = 0;
vBound = Lind;
exactValueEst  = Lind;
newErmBound = Lind;
numAlgs = 10000;
R = [];

for L = Lind
L
    cnt = cnt + 1;
l = L/2;

%Генерация выборки
[sample, sampleClasses] = GenerateSimpleSample(L)
 %[sample, sampleClasses] = GenerateRandomSample(L, 2);
%R = BuildLinearSet(sample, sampleClasses);
%Построение семейства алгоритмов по выборке

 R = BuildLinearSet(sample, sampleClasses);

%scatter(sample(:, 1), sample(:, 2), 10, sampleClasses, 'filled');
%hold off


%R = BuildAlgorithmSet(L, 0, numAlgs, 4);
numAlgs = size(R, 1);
%PaintAlgorithmsFamily(R);



%Вычисление разных типов оценок для семейства алгоритмов

    exactValueEst = MonteCarloEstimation(R, l, eps);
    vBound = VapnikFunctional(R, l, eps);
    newErmBound = NewERMFunctional(R, l, eps);
    uniformBound = Connected

% temp = MonteCarloEstimation(R, l, eps);
% exactValueEst(cnt) = temp(numAlgs);
% temp = VapnikFunctional(R, l, eps);
% vBound(cnt) = temp(numAlgs);
% temp = NewERMFunctional(R, l, eps);
% newErmBound(cnt) = temp(numAlgs); 


end
figure
hold on
plot ([1:numAlgs], vBound, 'g');
plot ([1:numAlgs], exactValueEst, 'r');
plot ([1:numAlgs], newErmBound, 'b');

% plot ([Lind], vBound, 'g');
% plot ([Lind], exactValueEst, 'r');
% plot ([Lind], newErmBound, 'b');
hold off

% % 
% % 
%  totalError = sum(R, 2);
%     
% D = zeros(1, L + 1);
%     
% for m = 0:L
%      D(m + 1) = sum(totalError == m);
% end
% D