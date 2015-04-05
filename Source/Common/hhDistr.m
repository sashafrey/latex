function h = hhDistr(L, l, m, s)
    h = 0;
    if (L >= 0 && l >= 0 && l <= L)
        for i = 0:s
            h = h + CnkCalc(m, i) * CnkCalc(L - m, l - i) / CnkCalc(L, l);
        end
    end
end
