function [cTable] = ComputeChooseTable(L)
%Вычисляет L + 1 уровень треугольника Паскаля 
	cTable = zeros(L + 1, L + 1);
    for i = 1:L+1
        cTable(i, 1) = 1;
        for j = 2:i 
            cTable(i, j) = cTable(i - 1, j - 1) + cTable(i - 1, j);
        end
    end
end