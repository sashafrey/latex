%Создает выборку
%L - Число объектов
%n - число признаков
%seed - инициализация генератора случайных чисел
%type = c(orrect), r(andom), n(oisy), m(odify correct)
% m = модификация корректной выборки в не корректную
function p=CreateSample(L,n,seed, varargin)
p.L = L;
p.n = 0;
p.pl = L/2/5;
if (p.pl<1),p.pl=1;end;
p.sample = [ones(L/2,1);zeros(L/2,1)];
p.rand = RandStream('mt19937ar','Seed',seed);
p.pAdd = 0;
p.nAdd = 0;

type = 'c';
if (~isempty(varargin))
    type = varargin{1};
end;

switch (type)
    case 'c'
        p = CreateCorrect1(p,n);
    case 'n'
        p = CreateNosiy1(p,n,1,20);
    case 'r'
        p = AddNoiseFeature(p,n);
    case 'm'
        p = CreateNosiySame(p, n, 10);
end;

p = NormalizeSample(p);
p = PrepareLogarithms(p);
end