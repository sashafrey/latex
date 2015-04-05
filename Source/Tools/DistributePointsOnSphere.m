function [x, vHistory, mindistHistory] = DistributePointsOnSphere(dim, N, params)
    % DISTRIBUTEPOINTSONSPHERE Evenly distributes N points on (dim - 1) dimentional sphere.
    % Treat each point as an electron constrained to a sphere, and run a
    % simulation for a certain number of steps.
    %
    % x = DISTRIBUTEPOINTSONSPHERE(m, n) returns the set of points where
    % rows correspond to points and columns corespond to dimentions.
    % 
    % [x, vHistory, mindistHistory] = DISTRIBUTEPOINTSONSPHERE(m, n)
	% returns history of points velocity and history of minimal distance.
    % Use plot(vNormHistory) or plot(mindistHistory) to produce historical
    % charts.
    %
    % Params: 
    %   k    - koef in Coulomb's low; default = 1.
    %   eta  - koef of viscosity; default = 0.3;
    %   dT   - delta for time iteratins; default = 0.05;
    %   vMin - min module of velocity, which defines stop criterion;
    %          default = 0.01
    %   maxIters - max number of iterations, default = 100;
    
    x = Initialize(N, dim);
    v = zeros(N, dim);
    
    if (~exist('params', 'var'))
        params = [];
    end
    
    params = SetDefault(params, 'vMin', 0.01);
    params = SetDefault(params, 'k', 1);
    params = SetDefault(params, 'eta', 0.3);
    params = SetDefault(params, 'dT', 0.05);
    params = SetDefault(params, 'maxIters', 100);

    vHistory = VectorCreate();
    mindistHistory = VectorCreate();
    iter = 0;
    while(true)
        [x, v] = Iterate(x, v, params.k, params.eta, params.dT);    
        vNorm = sqrt(sum(v .* v, 2));
        vHistory = VectorAdd(vHistory, vNorm');
        mindistHistory = VectorAdd(mindistHistory, CalcMinDistance(x));
        if (max(vNorm) < params.vMin)
            break;
        end
        
        %t = x(x(:, 3) > 0, :);
        %plot(t(:, 1), t(:, 2), '.');
        %axis([-1, 1, -1, 1]);
        %pause(0.01);
        
        iter = iter + 1;
        if (iter > params.maxIters)
            break;
        end
    end
    
    vHistory = VectorTrim(vHistory);
    vHistory = vHistory.Data;
    mindistHistory = VectorTrim(mindistHistory);
    mindistHistory = mindistHistory.Data;
end

function x = Initialize(N, dim)
    x = VectorCreate();
    iRemains = N;
    while (iRemains > 0)
        %r = rand(N, dim) ;
        r = 2 * (rand(N, dim) - 0.5);
        rNorm = sum(r .* r, 2);
        
        %Option one: throw away points with norm >1 to have ponts uniformly
        %distributed on sphere.
        %inSphere = rNorm <= 1;
        
        %Option two: don't throw away points, because in high dimentional
        %space the volume of the ball is too small comparing to the volume
        %of the cube.
        inSphere = true(N, 1); 
        
        r = r(inSphere, :);
        rNorm = sqrt(rNorm(inSphere));
        r = bsxfun(@rdivide, r, rNorm);
        
        if (size(r, 1) > iRemains)
            r = r(1:iRemains, :);
        end
        
        x = VectorAdd(x, r);
        iRemains = iRemains - size(r, 1);
    end
    
    x = VectorTrim(x);
    x = x.Data;
end

function [x, v] = Iterate(x, v, k, eta, dT)
    N = size(x, 1);
    dim = size(x, 2);
    
    f = zeros(N, dim);
    for i=1:N
        for j=1:N
            if (i==j)
                continue;
            end
            
            x1 = x(i, :);
            x2 = x(j, :);
            a = x1 - x2;
            f12 = k * a / power(a * a', 3/2);
            f12 = f12 - (f12 * x1') * x1;            
            f(i, :) = f(i, :) + f12;
        end
    end
    
    v = (1 - eta) * v + f * dT;
    x = x + v * dT;
    
    xNorm = sqrt(sum(x .* x, 2));
    x = bsxfun(@rdivide, x, xNorm);    
end

function minDistance = CalcMinDistance(x)
    N = size(x, 1);
    minDistance = +Inf;
    for i=1:N
        for j=(i+1):N
            x1 = x(i, :);
            x2 = x(j, :);
            a = x1 - x2;
            aDist = sqrt(a * a');
            if (aDist < minDistance)
                minDistance = aDist;
            end
        end
    end    
end