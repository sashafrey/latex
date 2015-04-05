function [profileErr, profileDeg] = estimateLinearScProfile(X, Y, iterCnt, varargin)
    L = size(X, 1);
    d = size(X, 2);
    algsCnt = computeNumberOfAlgs(L, d);
    
    display = true;
    if (length(varargin) == 1 && varargin{1} == false)
        display = false;
    end
    
%     algs_levels = zeros(1, iterCnt);
%     algs_bounds = zeros(1, iterCnt);
    
    binCoef = zeros(L + 1, L + 1);
    for i = 0:L
        for j = 0:L
            binCoef(i + 1, j + 1) = cnk(i, j);
        end
    end
    
    X = [X ones(L, 1)];
    profileErr = zeros(L + 1, 1);
    profileDeg = zeros(1, L + 1);
    for iter = 1:iterCnt
        if (display && mod(iter, 10) == 0)
            fprintf('%d\n', iter);
        end
        
        descr = zeros(1, d);
        for i = 1:d
            if i == 1
                descr(i) = randi(L - (d - i));
            else
                t = randi(L - (d - i) - descr(i - 1));
                descr(i) = t + descr(i - 1);
            end
        end
        
        A = X(descr, :);
        A(d + 1, 1:(d + 1)) = 0;
        A(d + 1, d + 1) = 1;
        b = zeros(d + 1, 1);
        if (rand(1) > 0.5)
            b(d + 1) = 1;
        else
            b(d + 1) = -1;
        end
        vertex = (A \ b)';
        errVect = zeros(1, L);
        for i = 1:L
            errVect(i) = (sign(X(i, :) * vertex') ~= Y(i));
        end
        errVect(descr) = 0.5 + sign(rand(1, d) - 0.5) / 2;
        
        %[upperNeighsCnt, lowerNeighsCnt] = getNeighboursFast(descr, vertex, errVect, X, Y);
        
        errCnt = sum(errVect);
        %deg = upperNeighsCnt;
        %connSetCnt = upperNeighsCnt + lowerNeighsCnt;
        
        profileErr(errCnt + 1) = profileErr(errCnt + 1) + 1;
        %profileErr(errCnt + 1) = profileErr(errCnt + 1) + (1 / algsCnt) / ((connSetCnt / nchoosek(L, d)) * 0.5 * (1 / 2^d));
        %profileDeg(deg + 1) = profileDeg(deg + 1) + (1 / algsCnt) / ((connSetCnt / nchoosek(L, d)) * 0.5 * (1 / 2^d));
        % а еще есть симметричный алгоритм
        %if (L - errCnt ~= errCnt)
        %    profileErr(L - errCnt + 1) = profileErr(L - errCnt + 1) + (1 / algsCnt) / ((connSetCnt / nchoosek(L, d)) * 0.5 * (1 / 2^d));
        %    profileDeg(lowerNeighsCnt + 1) = profileDeg(lowerNeighsCnt + 1) + (1 / algsCnt) / ((connSetCnt / nchoosek(L, d)) * 0.5 * (1 / 2^d));
        %end
        
        % вычисляем вклад в оценку
%         m = errCnt;
%         u = upperNeighsCnt;
%         bound = Inf;
%         for j = 1:sourcesCnt
%             a = sum(errVect < sources(j, :));
%             b = sum(errVect > sources(j, :));
%             currBound = 0;
%             for t = 0:min(a, b)
%                 currBound = currBound + ((my_cnk(b, t, binCoef) * ...
%                     my_cnk(L - u - b, l - u - t, binCoef)) / my_cnk(L, l, binCoef)) * ...
%                     my_hhDistr(L - u - b, l - u - t, m - b, floor((l/L) * (m - eps * (L - l))), binCoef);
%             end
%             bound = min(bound, currBound);
%         end
%         algs_levels(iter) = m;
%         algs_bounds(iter) = bound;
    end
    
    profileErr = profileErr * (algsCnt / iterCnt);
    %profileDeg = profileDeg * (algsCnt / iterCnt);
    %profileDeg = profileDeg / sum(profileDeg);
end

function algsCnt = computeNumberOfAlgs(L, d)
    algsCnt = 0;
    for i = 0:d
        algsCnt = algsCnt + nchoosek(L - 1, i);
    end
    algsCnt = 2 * algsCnt;
end

function res = my_cnk(n, k, binCoef)
    if (n >= 0 && k >= 0 && k <= n)
        res = binCoef(n + 1, k + 1);
    else
        res = 0;
    end
end

function h = my_hhDistr(L, l, m, s, binCoef)
    h = 0;
    if (L >= 0 && l >= 0 && l <= L)
        for i = 0:s
            h = h + my_cnk(m, i, binCoef) * my_cnk(L - m, l - i, binCoef) / my_cnk(L, l, binCoef);
        end
    end
end