function [Q, P] = CalcQEpsHasseByType(Prepare, eps, ell, k)
    % without upper connectivity.
    
    % Prepare must have q_total, m_etype, ma_etype. For description see
    % comments inside PrepareQEpsHasseByType.
    
    clustering = isfield(Prepare, 'mr_etype') & isfield(Prepare, 'r_etype') & isfield(Prepare, 'd') & isfield(Prepare, 'NSO');
    
    L = ell + k;
    
    nAlgs = length(Prepare);
    
    CLl = CnkCalc(L, ell);
    
    %������ ������ � ��������, ������������ ���� "1".
    Q = 0;
    P = 0;
    for i = 1:nAlgs
        %����� ���� ������ Qeps � ������ ������ ������� � ������� ����. 
        %��. ������� 12.2 (������ ����������-��������� ������ 1 � 2-��� ����)
        La = L - Prepare(i).q_total;
        curP = CnkCalc(La, ell) / CLl;
        
        if (clustering)
            curH = NoisedSetOverfittingCalc(Prepare(i).NSO, eps);
        else
            s_eps = floor(ell / L * (Prepare(i).m_etype - eps * k));
            curH = hhDistr(La, ell, Prepare(i).m_etype - Prepare(i).q_etype, s_eps);
        end

        Q = Q + curP * curH;
        P = P + curP;
    end
end