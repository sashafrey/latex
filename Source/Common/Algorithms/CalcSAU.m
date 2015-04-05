function [ sau, sau_storage ] = CalcSAU( sau_args, sau_storage)
    % sau_storage must be a SortedMatrix object
    % sau_args(2:end) is sorted in descending order.
    % also, the following requirement must hole:  length(sau_args) <= size(sau_storage.Data, 2) 
    % u = sau_args(1), restrictions v = sau_args(2:end)
    [contains, index] = SortedMatrixContains(sau_storage, sau_args);
    if (contains)
        sau = SortedMatrixGet(sau_storage, index);        
        sau = sau(end);
    else
        u = sau_args(1);
        v1 = sau_args(2);
        
        if (u == 0)
            sau = 1;
            return;
        end

        % from here we assume u~=0.
        if (v1 == 0)
            sau = 0;
            return;
        end       
        
        sau = 0;
        for i = 0 : min(u, v1)
            [cur_sau, sau_storage] = CalcSAU([u - i, sau_args(3:end), 0], sau_storage);
            sau = sau + cur_sau;
        end
        
        sau_storage = SortedMatrixAdd(sau_storage, sau_args, sau);
    end
end

