function [avgTrainErrors, avgTestErrors] = LibSVM_Evaluation(indexes, task, params)
    avgTrainError = 0;
    avgTestError = 0;

    task.objects(isnan(task.objects)) = 0.0;
    
    qFolds = length(unique(indexes));
    for foldIter = 1:qFolds
        trainTask = GetTaskSubsample(task, indexes ~= foldIter);
        testTask  = GetTaskSubsample(task, indexes == foldIter);

        [composition] = svmtrain2(trainTask.target, trainTask.objects, params.LibSvmTrainOptions);
        trainPrediction = svmpredict(trainTask.target, trainTask.objects, composition, params.LibSvmPredictOptions);
        testPrediction = svmpredict(testTask.target, testTask.objects, composition, params.LibSvmPredictOptions);

        avgTrainError = avgTrainError + sum(trainPrediction ~= trainTask.target);
        avgTestError = avgTestError + sum(testPrediction ~= testTask.target);
    end

    avgTrainErrors = avgTrainError / (qFolds * task.nItems);
    avgTestErrors = avgTestError / task.nItems;
end
