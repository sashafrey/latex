function [avgTrainErrors, avgTestErrors] = RVM_Evaluation(indexes, task, params)
    avgTrainError = 0;
    avgTestError = 0;

    task.objects(isnan(task.objects)) = 0.0;
    
    qFolds = length(unique(indexes));
    for foldIter = 1:qFolds
        trainTask = GetTaskSubsample(task, indexes ~= foldIter);
        testTask  = GetTaskSubsample(task, indexes == foldIter);

        [composition] = rvmtrain(trainTask, params);
        trainPrediction = rvmpredict(trainTask, composition);
        testPrediction = rvmpredict(testTask, composition);

        avgTrainError = avgTrainError + sum(trainPrediction ~= trainTask.target);
        avgTestError = avgTestError + sum(testPrediction ~= testTask.target);
    end

    avgTrainErrors = avgTrainError / (qFolds * task.nItems);
    avgTestErrors = avgTestError / task.nItems;
end
