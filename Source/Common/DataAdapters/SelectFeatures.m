function subsample = SelectFeatures(sample, featureIds)
    % Selects subset of features from the task
    if (islogical(featureIds))
        subsample.nFeatures = sum(featureIds);
    else
        subsample.nFeatures = length(featureIds);
    end
    
    subsample.nItems = sample.nItems;
    subsample.nClasses = sample.nClasses;
    subsample.target = sample.target;
    subsample.objects = sample.objects(:, featureIds);
    subsample.isnominal = sample.isnominal(featureIds);
    subsample.name = sample.name;
    
    if (isfield(sample, 'filename'))
        subsample.filename = sample.filename;
    end
end
