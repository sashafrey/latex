%������� ������� ������ �� ���������� ������� ����-���� ������������.
%�������� ������ �������� (� ������ � �� ������), ������ �� ��������� �
%������� ��� ��� �� ���������� ������������.

function p = CreateNosiySame(p, n, v)
p = CreateCorrectN(p,n);%�������� ���������� �������.

iC1 = find(p.sample(:,end)==1);
iC0 = find(p.sample(:,end)==0);
for i=1:v
    id1 = p.rand.randi(length(iC1));
    id0 = p.rand.randi(length(iC0));
    tmp = p.sample(id1,2);
    p.sample(id1,2) = p.sample(id0,2);
    p.sample(id0,2) = tmp;
    iC1(id1) = [];
    iC0(id0) = [];
end;

end