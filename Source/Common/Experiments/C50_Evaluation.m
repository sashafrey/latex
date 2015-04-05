function [avgTrainErrors, avgTestErrors] = C50_Evaluation(indexes, task, params)
    avgTrainError = 0;
    avgTestError = 0;

    task = C50Prepare(task);    

    qFolds = length(unique(indexes));
    for foldIter = 1:qFolds
        trainTask = GetTaskSubsample(task, indexes ~= foldIter);
        testTask  = GetTaskSubsample(task, indexes == foldIter);

        [composition] = C50Tune(trainTask, params);
        trainPrediction = C50Calc(composition, trainTask);
        testPrediction = C50Calc(composition, testTask);

        avgTrainError = avgTrainError + sum(trainPrediction ~= trainTask.target);
        avgTestError = avgTestError + sum(testPrediction ~= testTask.target);
    end

    avgTrainErrors = avgTrainError / (qFolds * task.nItems);
    avgTestErrors = avgTestError / task.nItems;

    C50Clean(task);
end
