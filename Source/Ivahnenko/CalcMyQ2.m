function [Qeps, Peps] = CalcMyQ2(p)

fixPoints = p.fixPoints;
l=p.L/2;
Qeps = zeros(2*l+1,2);
eps = -1:1/l:1;
Qeps(:,1) = eps;
%�����������
Peps = zeros(length(p.errors),1);
minErrors = double(min(p.errorsCount));
razb = nchoosek(p.L,l);
% ������ ������� ��� ������� ��������� ��� ����� � ����� Qeps
for aIdx=1:length(p.errors)
    pTrain = fixPoints{aIdx,1};
    pTest = fixPoints{aIdx,2};
    if (isnan(pTrain)),continue;end;
    % ������� ������� �������� ������ �� �������� |X_av|
    fpTrain = length(pTrain);
    % ������� ������� �������� ������ �� �������� |X'_av|
    fpTest = length(pTest);
    %������� ��� ��������� ������ �� ��������.
    nTrain = length(intersect(pTrain, p.errors{aIdx}));
    %������� ��� ��������� ������ �� ��������.
    nTest = length(intersect(pTest, p.errors{aIdx}));
    
    L_a = p.L - fpTrain-fpTest;%������� �������� ���������������� ��������
    l_a = l - fpTrain;%������� �������� ���� �� ��������
    
    %����� ������ ������� �� ��� ����� ���������
    m_a = double(p.errorsCount(aIdx)-nTest-nTrain);
    if (L_a==p.L || l_a<0 || L_a < 0 || l_a>L_a || L_a-m_a<0)
        % l_a <0 - ���������� ����� �������� ������ ������� � ���� ���� ���
        % ��� ����������� ��������� ������ ������� ��������� �������
        continue; 
    end;
        
%     razb_a = nchoosek(L_a, l_a);%������� ����� �������� ���������� ��������� ���������� ��������
%     P_a = razb_a/razb;%"�����������" ������ ����� ���������. �������� ��������!
%     Peps(aIdx) = Peps(aIdx) + razb_a;
%     s_a = p.l/p.L*(double(p.algErrors(aIdx,1))-eps*(p.L-p.l))-(nTrain);
%     h = H(s_a, L_a, m_a, l_a);
%     Qeps(:,2) = Qeps(:,2)+P_a*h;
    
%������: ��� � ������ � �������� ������� �������?
%1) min(minErrors-nTrain, m_a)
%2) min(m_a, l_a)
    s0 = max(0, m_a-(L_a-l_a));
    s_a = l/p.L*(p.errorsCount(aIdx)-eps*(p.L-l))-nTrain;
    for s=s0:min([minErrors-nTrain, m_a, l_a])
        if (l_a-s<0 || L_a-m_a<l_a-s), continue;end;
        razb_a = nchoosek(m_a, s);
        razb_a = razb_a * nchoosek(L_a-m_a, l_a-s);
        P_a = razb_a/razb;%"�����������" ������ ����� ���������. �������� ��������!
        Peps(aIdx) = Peps(aIdx) + razb_a;

        idx = find(s_a>s, 1, 'last' )+1;
        if(isempty(idx)), break; end;
        Qeps(idx,2) = Qeps(idx,2)+P_a;
    end;
end;

end
