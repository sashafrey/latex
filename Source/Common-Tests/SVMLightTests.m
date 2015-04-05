function SVMLightTests()
    task = LoadTask('wine');
    
    [trainTask, testTask] = SplitTask(task, 0.05);
    svmClassifier = svmlearn(trainTask.objects, 2 * trainTask.target - 3, '-t 0 -c 0.5 -v 0');
    [avgTrainError, trainPrediction] = svmclassify(trainTask.objects, 2 * trainTask.target - 3, svmClassifier);
    [avgTestError, testPrediction]  = svmclassify(testTask.objects, 2 * testTask.target - 3, svmClassifier);

    Check(avgTrainError < 0.35);
    Check(avgTestError < 0.35);
end