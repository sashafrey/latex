function dir_all = getRandomDirections(dirCnt, dim)
% возвращает матрицу размера (dirCnt * dim),
% каждая строка которой - случайно выбранный вектор

    dir_all = rand(dirCnt, dim) - 0.5;
end