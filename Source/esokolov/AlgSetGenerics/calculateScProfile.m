function Profile = calculateScProfile(A, L)
    fprintf('Calculating profile...\n');
    %Profile = zeros(L + 1);
    Profile = zeros([L+1, L+1, L+1]);
    for i = 1:size(A, 1)
        if (mod(i, 100) == 0)
            fprintf('%d\n', i);
        end
        %m = length(find(A(i, :)));
        m = sum(A(i, :));
        q = 0;
        r = 0;
        a = A(i, :);
        for j = 1:size(A, 2)
            if (a(j) == 0)
                a(j) = 1;
%                 for z = 1:size(A, 1)
%                     if (A(z, :) == a)
%                         q = q + 1;
%                         break;
%                     end
%                 end
                if (binarySearch(A, a) == 1)
                   q = q + 1;
                end
                a(j) = 0;
            end
        end
        
        rm = zeros(1, L + 1);
        for j = 1:size(A, 1)
            if (i == j)
                continue;
            end
            
            errCnt = 0;
            good = 1;
            for k = 1:size(A, 2)
                if (A(j, k) == 1 && A(i, k) == 0)
                    good = 0;
                    break;
                end
                if (A(j, k) == 1)
                    errCnt = errCnt + 1;
                end
            end
            if (good && errCnt < m)
                for k = 1:size(A, 2)
                    if (A(j, k) == 0 && A(i, k) == 1)
                        rm(k) = 1;
                    end
                end
            end
        end
        r = sum(rm);

        Profile(m + 1, q + 1, r + 1) = Profile(m + 1, q + 1, r + 1) + 1;
        %Profile(m + 1, q + 1) = Profile(m + 1, q + 1) + 1;
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
            res = 1;
            break;
        end
    end
end
