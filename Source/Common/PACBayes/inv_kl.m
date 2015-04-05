function x = inv_kl(p, y, eps)
    % INV_KL is an 'inverse' of KL function. 
    %
    % x = INV_KL(p, y) gives the largest value 'x', such as KL(p, x) <= y.
    % x = INV_KL(p, y, eps) allows to specify the precision. Default:0.001.
    
    if (~exist('eps', 'var'))
        eps = 0.001;        
    end
    
    % ToDo: binary search or Newton method might be more efficient here.
    xx  = 0:eps:1;
    id = find(kl(p, xx) <= y, 1, 'last');
    if (isempty(id))
        x = NaN;
    else
        x = xx(id);
    end
end