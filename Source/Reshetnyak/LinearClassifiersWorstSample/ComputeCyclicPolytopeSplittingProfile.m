function [profile] = ComputeCyclicPolytopeSplittingProfile(numObjects, dim, cTable)
    if nargin < 3
        cTable = ComputeChooseTable(numObjects);
    end
    profile = zeros(1, numObjects + 1);
    profile(1) = 1;
    profile(numObjects + 1) = 1;
    n = numObjects;
    for k = 1 : numObjects - 1
        for s = 1 : dim
            q = floor((s + 1)/2);
            if mod(s, 2) == 0
                profile(k + 1) = profile(k + 1) + cTable(k, q + 1) * cTable(n - k, q) + cTable(k, q) * cTable(n - k, q + 1); 
            else
                profile(k + 1) = profile(k + 1) + 2 * cTable(k, q) * cTable(n - k, q);
            end
        end
    end
end