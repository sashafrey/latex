function outSample = ConvertNominalFeatures(inSample)
    outSample.nItems = inSample.nItems;
    outSample.nClasses = inSample.nClasses;
    outSample.name = inSample.name;
    outSample.target = inSample.target;
    f = length(find(outSample.target==outSample.target(1)))/outSample.nItems;
    idx = find(inSample.isnominal)';
    s=0;
    for i=idx
        vals = unique(inSample.objects(:,i));
        info{i} = vals';
        if (sum(isnan(inSample.objects(:,i)))>0)
            info{i} = [info{i}, NaN];
        end;
        s = s+length(vals);
    end;
    outSample.objects = [inSample.objects(:, find(~inSample.isnominal)'), ones(inSample.nItems, s)*f];
    outSample.nFeatures = sum(~inSample.isnominal);
    for i=idx
        for j=info{i}
            outSample.nFeatures = outSample.nFeatures + 1;
            if (isnan(j))
                outSample.objects(isnan(inSample.objects(:, i)), outSample.nFeatures) = (1-f); 
            else
                outSample.objects(inSample.objects(:, i)==j, outSample.nFeatures) = (1-f); 
            end;
        end;
    end;
    
    outSample.isnominal = zeros(outSample.nFeatures,1);
end