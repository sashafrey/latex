%parm - структура в окторой лежит таблица логарифмов
%Pos - общее число своих объектов
%p - число покрытых своих объектов
%Neg - общее число чужих объектов
%n - число покрытых чужих объектов
function info=CalcInfo(parm, Pos, p, Neg, n)
p = round(max(1,min(p,Pos)));
n = round(max(1,min(n,Neg)));
info = parm.logs(Neg+1) -  parm.logs(n+1) -  parm.logs(Neg - n+1) + ...
		parm.logs(Pos+1) -  parm.logs(p+1) -  parm.logs(Pos - p+1) - ...
		parm.logs(Neg+Pos+1) + parm.logs(n+p+1) +  parm.logs(Neg+Pos-n-p+1);
if (p>n)
    info = -info;
end;
end