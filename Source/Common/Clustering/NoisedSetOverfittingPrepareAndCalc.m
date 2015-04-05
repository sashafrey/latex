function [PrepareNSO, QEps] = NoisedSetOverfittingPrepareAndCalc(L, ell, m1, m_r, r, d, eps)
    PrepareNSO = NoisedSetOverfittingPrepare(L, ell, m1, m_r, r, d);
    QEps = NoisedSetOverfittingCalc(PrepareNSO, eps);
end
