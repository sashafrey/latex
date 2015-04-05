function LibSVMTests
    task = LoadTask('wine');
    
    [trainTask, testTask] = SplitTask(task, 0.5);

    svmClassifier = svmtrain2(trainTask.target, trainTask.objects, '-q');
    trainPrediction = svmpredict(trainTask.target, trainTask.objects, svmClassifier, '-q');
    testPrediction  = svmpredict(testTask.target, testTask.objects, svmClassifier, '-q');

    avgTrainError = sum(trainPrediction ~= trainTask.target) / trainTask.nItems;
    avgTestError = sum(testPrediction ~= testTask.target) / testTask.nItems;
    
    Check(avgTrainError < 0.35);
    Check(avgTestError < 0.35);
end