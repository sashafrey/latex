function [avgTrainErrors, avgTestErrors] = COMBoost_Evaluation(indexes, task, params)
    avgTrainError = 0;
    avgTestError = 0;

    qFolds = length(unique(indexes));
    for foldIter = 1:qFolds
        trainTask = GetTaskSubsample(task, indexes ~= foldIter);
        testTask  = GetTaskSubsample(task, indexes == foldIter);
        terms = Calibrate(trainTask);

        if (params.adjust)
            params2 = params;
            %params.calcHasse = true;
            params.calcCV = true;
            params.nItersCV = 1000;
            params.parallel = false;
            params.verbose = false;
            params.fAdjust = @AdjustPNpnVoid;
            adjTable = CreateAdjustmentsTable(trainTask, params);
            %adjTable = adjTable.Hasse;
            adjTable = adjTable.CV;
            params = params2;           
            
            params.fAdjust = @(P, N, p, n, rules)AdjustPNpn(P, N, p, n, rules, adjTable);
        end
        
        [composition] = COMBoost(trainTask, terms, params);

        trainPrediction = ClassifyComposition(composition, terms, trainTask, 1);
        testPrediction = ClassifyComposition(composition, terms, testTask, 1);
        avgTrainError = avgTrainError + sum(trainPrediction ~= trainTask.target);
        avgTestError = avgTestError + sum(testPrediction ~= testTask.target);
    end
    
    avgTrainErrors = avgTrainError / (qFolds * task.nItems);
    avgTestErrors = avgTestError / task.nItems;
end
