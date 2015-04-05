function task = NormalizeFeatures( task, convertTarget, normalizeType )
    if (~exist('convertTarget', 'var'))
        convertTarget = false;
    end
    
    if (~exist('normalizeType', 'var'))
        normalizeType = 1;
    end
    
    if (convertTarget)
        Check(all(unique(task.target)' == [1, 2]));
        
        % to make it easy for SVM
        task.target = 2 * (task.target - 1.5);
    end
    
    for i = 1:task.nFeatures
        m  = mean(task.objects(:, i));
        st = std(task.objects(:, i));
        
        minval = min(task.objects(:, i));
        maxval = max(task.objects(:, i));

        switch (normalizeType)
          case 1
            Check(~isnan(m));
            Check(~isnan(st));
          	if (st < 1e-5) 
            	continue;
            end

            task.objects(:, i) = (task.objects(:, i) - m) / st;
          case 2
            Check(~isnan(minval));
            Check(~isnan(maxval));
            if (maxval - minval > 1e-7)
                task.objects(:, i) = (task.objects(:, i) - minval) / (maxval - minval);
            else
                task.objects(:, i) = (task.objects(:, i) - minval);
            end             
        end
    end    
end

