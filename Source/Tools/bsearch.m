function x = bsearch(f, yTarget, x0, eps, maxValue)
    if (~exist('eps', 'var'))
        eps = 0.01;
    end
    
    if (~exist('maxValue', 'var'))
        maxValue = 1;
    end
    
    % assumes that f is decreasing function.
    x = x0;

    xMin = NaN;
    xMax = NaN;
    y = f(x);
    
    if (y >= yTarget) xMin = x; end;
    if (y <= yTarget) xMax = x; end;
    

    while(isnan(xMin))
        xMax = x;
        x = x / 2;
        if (x < eps)
            return;
        end
        
        y = f(x);
        if (y >= yTarget)
            xMin = x;
            break;
        end
    end
    
    while(isnan(xMax))
        xMin = x;
        x = x * 2;
        
        y = f(x);
        if (y <= yTarget)
            xMax = x;
            break;
        end        
        
        if (x > maxValue)
            x = maxValue;
            return;            
        end
    end
    
    while((xMax - xMin) > eps)
        x = (xMax + xMin) / 2;
        y = f(x);
        if (y <= yTarget)
            xMax = x;
        else
            xMin = x;
        end
    end
end