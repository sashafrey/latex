function [alg, neighbours] = getNeighboursLC_rays(alg, X, Y, raysCnt)
% ���������� ��� ��������� alg ���� �������, ��������� ������� ���������
% �����������.

    [N, d] = size(X);
    eps = 1e-6;
    
    neighbours = StructVectorCreate(initLinearAlgSimpleStructure());
    
    dir_all = getRandomDirections(raysCnt, d);
    
    for currRayNum = 1:raysCnt
        % ���������� ����������� ����
        dir = dir_all(currRayNum, :)';
        
        % ����������� ���� � ����������������
        intersections = inf(N, 1);
        
        % TODO: ������������� ����
        for currHyperplane = 1:N
            w_curr = X(currHyperplane, :)';
            % ���� ��� ���������� ��������������, �� ���������� ��
            if abs(w_curr' * dir) < eps
                continue;
            end
            
            intersections(currHyperplane) = ...
                - (w_curr' * alg.w) / (w_curr' * dir);
        end
        
        % � �� ��������� �� �� ��� � �������?
        % ���� ��, �� ������������� ���
        if sum(intersections > 0) <= 1
            dir = -dir;
            intersections = -intersections;
        end
        
        % ���� ����������� �������� ������ ����, �� ��� �� ����������
        % ��������������; ����������� ����� �����������
        intersections(intersections <= 0) = Inf;
        
        intersect_sorted = sort(intersections);
        coef_new = (intersect_sorted(1) + intersect_sorted(2)) / 2;
        w_new = alg.w + coef_new * dir;
        
        alg_new = convertLinearAlgToSimpleStructure(w_new, X, Y);
        
        % �����, ����� �������� ��� ���� � neighbours?
        is_new = true;
        for neighIdx = 1:StructVectorLength(neighbours)
            if isEqualLC(neighbours.Data(neighIdx), alg_new)
                is_new = false;
                break;
            end
        end
        
        % ���� ������ �� ��� �� ������, �� ������ ���������!
        if is_new
            neighbours = StructVectorAdd(neighbours, alg_new);
            
            if alg_new.errCnt < alg.errCnt
                alg.lowerNeighsCnt = alg.lowerNeighsCnt + 1;
            else
                alg.upperNeighsCnt = alg.upperNeighsCnt + 1;
            end
        end
    end    
end
