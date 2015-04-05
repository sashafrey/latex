function C50Tests
    task = LoadTask('german');
    task = C50Prepare(task);
    
    [trainTask, testTask] = SplitTask(task, 0.5);
    [composition] = C50Tune(trainTask);
    trainPrediction = C50Calc(composition, trainTask);
    testPrediction = C50Calc(composition, testTask);

    avgTrainError = sum(trainPrediction ~= trainTask.target) / trainTask.nItems;
    avgTestError = sum(testPrediction ~= testTask.target) / testTask.nItems;
    
    Check(avgTrainError < 0.35);
    Check(avgTestError < 0.35);

    C50Clean(task);
end