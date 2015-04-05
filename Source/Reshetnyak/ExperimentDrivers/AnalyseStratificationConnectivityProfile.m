sampleSize = 20;
eps = 0.08;
trainSize = sampleSize / 2;
%[sample, sampleClasses] = GenerateCloseClasses(sampleSize);
[sample, sampleClasses] = GenerateSimpleSample(sampleSize, 6);
%[sample, sampleClasses] = GenerateConvexPolytope(sampleSize, 3);
%[algs, familyGraph] = BuildLinearSet(sample, sampleClasses);
familyGraph = BuildLinearSet(sample, sampleClasses);
scProfile = ComputeScProfile(1, familyGraph);
   
%pmqTable = ComputePmqTable(sampleSize, trainSize, eps, 0, size(scProfile, 2));
% 	levelsBound = sum( pmqTable(1:size(scProfile,1), 1:size(scProfile, 2)) .* scProfile, 2);
%PaintSample(sample, sampleClasses);
%PaintAlgorithmsFamily(algs, 4);
%figure
%bar3(scProfile', 1, 'detached');
%olormap(1 - 0.8*gray);
%graymon
%surf(scProfile);
%surf(0:size(scProfile, 2) - 1, 0:(size(scProfile, 1) - 1), scProfile);
%xlabel('q');
%ylabel('m');
%zlabel('D_{mq}');
plot(0 : size(scProfile, 2) - 1, sum(scProfile, 1));
grid on
sum(scProfile, 1)
% 	figure
% 	bar(levelsBound(2:sampleSize+1), 1, 'grouped');
% 	colormap('default');
%     
%     sumLevelsBound = levelsBound;
%     for n = 2:numel(sumLevelsBound)
%         sumLevelsBound(n) = sumLevelsBound(n - 1) + levelsBound(n);
%     end
%     figure
%     plot([0:sampleSize], sumLevelsBound);
% 	
