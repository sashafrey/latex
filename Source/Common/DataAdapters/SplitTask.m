function [trainTask, testTask, trainIds, testIds] = SplitTask(task, trainRatio)
    ell = floor(task.nItems * trainRatio);
    trainIds = randsampleStratified(task.target, ell);
    ids = true(task.nItems, 1);
    ids(trainIds) = false;
    testIds = find(ids);
    
    trainTask = GetTaskSubsample(task, trainIds);
    testTask = GetTaskSubsample(task, testIds);    
end
