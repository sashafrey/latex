function subsample = GetTaskSubsample(sample, ids)
    if (islogical(ids))
        subsample.nItems = sum(ids);
    else
        subsample.nItems = length(ids);
    end
    
    subsample.nFeatures = sample.nFeatures;
    subsample.nClasses = sample.nClasses;
    subsample.target = sample.target(ids);
    subsample.objects = sample.objects(ids, :);
    subsample.isnominal = sample.isnominal;
    subsample.name = sample.name;
    
    if (isfield(sample, 'filename'))
        subsample.filename = sample.filename;
    end
end
