function outSample = GenerateNewFeatures(inSample)
    outSample.nItems = inSample.nItems;
    outSample.nClasses = inSample.nClasses;
    outSample.name = inSample.name;
    outSample.target = inSample.target;
    outSample.nFeatures = inSample.nFeatures;
    outSample.objects = [inSample.objects, zeros(inSample.nItems, inSample.nFeatures*(inSample.nFeatures+1))];
    for i=1:inSample.nFeatures
        for j=i:inSample.nFeatures
            outSample.nFeatures = outSample.nFeatures + 1;
            outSample.objects(:, outSample.nFeatures) = inSample.objects(:,i) .* inSample.objects(:,j); 
        end;
    end;
    
    outSample.isnominal = zeros(outSample.nFeatures,1);
end