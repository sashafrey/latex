function h = hhProb(L, l, m, i)
    h = 0;
    if (L >= 0 && l >= 0 && l <= L)
        h = CnkCalc(m, i) * CnkCalc(L - m, l - i) / CnkCalc(L, l);
    end
end
