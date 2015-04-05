function [P1, P11] = PrepareQEpsHasseByType(errorVectors, edges, target, algClasses )
    % without upper connectivity.
    % target - ������ ������� ��������
    % algCalss - ������ �� ������� �������� ���������
    L = size(errorVectors,2);
    
    nAlgs = size(errorVectors,1);
    assert(L==length(target));
    assert(length(algClasses)==nAlgs);

    s.q_total = 0;  % In both P1 and P11 q_total stores the same value - the total "definiency" of algorithm.
    s.m_etype = 0;  % "etype" = "error type". The m_etype stores the number of errors of a given type (1 or 11)
    s.q_etype = 0;  % "q_etype" = the deficiency of an algorithm for a given error type (1 or 11)
    
    P1 = repmat(s, nAlgs, 1);
    P11 = repmat(s, nAlgs, 1);
    
    %������ ������ � ��������, ������������ ���� "1".
    for i = 1:nAlgs
        algEV = errorVectors(i,:);
        
        if (~isempty(edges))
            parentsIntersection = true(1, L);
            allParents = GetAllParents(edges, i);
            for j=1:length(allParents)%TODO: �������� �� ������ �� ������� ��������
                parent = errorVectors(allParents(j),:);
                parentsIntersection = (parentsIntersection & parent);
            end
            assert(all(parentsIntersection <= algEV));
        end
        
        %����� ���� ������ Qeps � ������ ������ ������� � ������� ����. 
        %��. ������� 12.2 (������ ����������-��������� ������ 1 � 2-��� ����)
        
        pos = (target == algClasses(i))';%������� ������ ������
        
        P1(i).m_etype = sum(algEV & pos);
        P11(i).m_etype = sum(algEV & ~pos);
        
        m_total = sum(algEV);
        
        if (~isempty(edges))
            P1_q_etype = P1(i).m_etype - sum(parentsIntersection & pos);
            P11_q_etype = P11(i).m_etype - sum(parentsIntersection & ~pos);
            q_total = m_total - sum(parentsIntersection);
        else
            P1_q_etype = 0;
            P11_q_etype = 0;
            q_total = 0;            
        end

        P1(i).q_total = q_total;
        P11(i).q_total = q_total;

        P1(i).q_etype = P1_q_etype;
        P11(i).q_etype = P11_q_etype;
    end
end