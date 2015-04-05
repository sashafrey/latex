function [ task ] = AddConstantFeature( task )
    task.objects = [task.objects, ones(task.nItems, 1)];
    task.nFeatures = task.nFeatures + 1;
end

