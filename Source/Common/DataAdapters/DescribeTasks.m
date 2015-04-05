function DescribeTasks( tasks )
    fields = fieldnames(tasks);
    fprintf('TaskName\t classes\t items\t numF\t catF\t gap%%\t\n');
    for i=1:length(fields)
        taskname = fields{i};
        task = tasks.(taskname);
        gapRatio = sum(sum(isnan(task.objects))) / (task.nItems * task.nFeatures);
        fprintf('%s\t%d\t%d\t%d\t%d\t%.1f\t\n', taskname, task.nClasses, task.nItems, sum(~task.isnominal), sum(task.isnominal), 100*gapRatio );
    end
end
