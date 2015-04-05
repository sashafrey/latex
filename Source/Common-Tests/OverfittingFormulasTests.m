function OverfittingFormulasTests
   Check(abs(BallSliceSubsetOverfitting(30, 15, 6, 4, 6) - 0.1338) < 0.001);
   
   L = 100;
   ell = 50;
   m1 = 20;
   m_r = 10;
   r = 5;
   d = 10;
   eps = 0.1;
   
   [PrepareNSO, QEps] = NoisedSetOverfittingPrepareAndCalc(L, ell, m1, m_r, r, d, eps);
   QEpsCalc = NoisedSetOverfittingCalc(PrepareNSO, eps);
   Check(abs(QEps - QEpsCalc) < 0.001);   
   
   QEpsEmpirical = NoisedSetOverfittingEmpirical(L, ell, m1, m_r, r, d, eps);
   Check(abs(QEpsEmpirical - 0.1819) < 0.15)
end