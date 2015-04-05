function RVMTests
    task = LoadTask('wine');
    
    [trainTask, testTask] = SplitTask(task, 0.5);

    rvmClassifier = rvmtrain(trainTask);
    trainPrediction = rvmpredict(trainTask, rvmClassifier);
    testPrediction  = rvmpredict(testTask, rvmClassifier);

    avgTrainError = sum(trainPrediction ~= trainTask.target) / trainTask.nItems;
    avgTestError = sum(testPrediction ~= testTask.target) / testTask.nItems;
    
    Check(avgTrainError < 0.35);
    Check(avgTestError < 0.35);
end