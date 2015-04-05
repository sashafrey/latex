function [A, hplanes, neighs] = getLinearAlgorithmsSet(Sample, Answers)
    L = size(Sample, 1);
    p = size(Sample, 2);
    
    A_maxSize = 500000;
    A = false(A_maxSize, L);
    hplanes = zeros(A_maxSize, p + 1);
    cnt = 0;
    fprintf('Generating combinations...\n');
    Combs = generateAllCombinations(L, p + 1);
    fprintf('%d combinations\n', size(Combs, 1));
    fprintf('Computing error matrix...\n');
    for i = 1:size(Combs, 1)
        if (mod(i, 1000) == 0)
            fprintf('%d\n', i);
        end
        
        if (cnt >= A_maxSize - 1)
            [Anew I] = unique(A(1:cnt, :), 'rows');
            cnt = size(Anew, 1);
            hplanes(1:cnt, :) = hplanes(I, :);
            A = false(A_maxSize, L);
            A(1:cnt, :) = Anew;
        end
        
        M = ones(p + 1);
        for j = 1:(p + 1)
            for k = 1:p
                M(j, k) = Sample(Combs(i, j), k);
            end
        end
        B = zeros(p + 1, 1);
        B(p + 1) = 1;
        w = linsolve(M, B);
        w0 = w(p + 1);
        w(p + 1) = [];
%         w = zeros(2, 1);
%         w(1) = Sample(Combs(i, 1), 2) - Sample(Combs(i, 2), 2);
%         w(2) = Sample(Combs(i, 2), 1) - Sample(Combs(i, 1), 1);
%         w0 = Sample(Combs(i, 1), 1) * Sample(Combs(i, 2), 2) - Sample(Combs(i, 2), 1) * Sample(Combs(i, 1), 2);
        
        coefs = [w; w0];
        a1 = zeros(1, L);
        a2 = a1;
        for j = 1:L
            if (isempty(find(Combs(i, 1:p) == j, 1)))
                class = sum(Sample(j, :) .* w') + w0;
                if (class >= 0)
                    class = 1;
                else
                    class = -1;
                end
                trueClass = Answers(j);
                if (class == trueClass)
                    a1(j) = 0;
                else
                    a1(j) = 1;
                end
                a2(j) = 1 - a1(j);
            end
        end
        
        for mask = 0:(bitshift(1, p + 1) - 1)
            for j = 1:p
                class = bitget(mask, j);
                a1(Combs(i, j)) = class;
                a2(Combs(i, j)) = 1 - class;
            end
            %A = [A; a1; a2];
            A(cnt + 1, :) = a1;
            A(cnt + 2, :) = a2;
            hplanes(cnt + 1, :) = coefs;
            hplanes(cnt + 2, :) = -coefs;
            cnt = cnt + 2;
        end
    end
    A((cnt + 1):end, :) = [];
    hplanes((cnt + 1):end, :) = [];
    fprintf('%d algorithms\n', size(A, 1));
    [A I] = unique(A, 'rows');
    hplanes = hplanes(I, :);
    fprintf('%d unique algorithms\n', size(A, 1));
    
    

    fprintf('Calculating profile...\n');
    algsCnt = size(A, 1);
    neighs(1:algsCnt) = struct('neighsCnt', 0, 'neighsList', zeros(1, 30));
    for i = 1:size(A, 1)
        if (mod(i, 100) == 0)
            fprintf('%d\n', i);
        end
        a = A(i, :);
        for j = 1:size(A, 2)
            a(j) = ~a(j);
            pos = binarySearch(A, a);
            if (pos >= 1)
               neighs(i).neighsCnt = neighs(i).neighsCnt + 1;
               neighs(i).neighsList(neighs(i).neighsCnt) = pos;
            end
            a(j) = ~a(j);
        end
    end
end

function res = binarySearch(A, el)
    res = 0;
    l = 1;
    r = size(A, 1);
    while (l <= r)
        m = floor((l + r) / 2);
        cmp = 0;
        for i = 1:size(A, 2)
            if (el(i) < A(m, i))
                cmp = -1;
                break;
            elseif (el(i) > A(m, i))
                cmp = 1;
                break;
            end
        end
        if (cmp < 0)
            r = m - 1;
        elseif (cmp > 0)
            l = m + 1;
        else
            res = m;
            break;
        end
    end
end