function dir_all = getRandomDirections(dirCnt, dim)
% ���������� ������� ������� (dirCnt * dim),
% ������ ������ ������� - �������� ��������� ������

    dir_all = rand(dirCnt, dim) - 0.5;
end