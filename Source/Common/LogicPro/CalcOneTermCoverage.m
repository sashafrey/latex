function coverage = CalcOneTermCoverage(left, right, feature, isnot, task)
    % term should be a stucture with the following fields:
    %   coverage, 
    %   feature, 
    %   left, 
	%   right, 
    %   isnot

    if (isnan(left) || isnan(right))
        coverage = isnan(task.objects(:, feature));
    elseif (left == right)
        coverage = task.objects(:, feature) == left;
    else 
        values = task.objects(:, feature);
        coverage = (values >= left) & (values < right);
    end
    
    if (isnot)
        coverage = ~coverage;            
    end
end