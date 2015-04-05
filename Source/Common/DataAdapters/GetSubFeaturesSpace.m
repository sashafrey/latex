function [ subsample ] = GetSubFeaturesSpace( sample, ids )
%GETSUBFEATURESSPACE Function that return same sample but with only some
%features.
%   sample - Sample to be subspaced
%   ids - indecies of features that will be in outSample

    if (islogical(ids))
        subsample.nFeatures = sum(ids);
    else
        subsample.nFeatures = length(ids);
    end
    
    subsample.nItems = sample.nItems;
    subsample.nClasses = sample.nClasses;
    subsample.target = sample.target;
    subsample.objects = sample.objects(:, ids);
    subsample.isnominal = sample.isnominal(ids);
    subsample.name = sample.name;
    
    if (isfield(sample, 'filename'))
        subsample.filename = sample.filename;
    end

end

