import numpy as np
import unittest
import LinearSampling.api as lsApi

__author__ = 'Administrator'


class TestAdvanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = lsApi.Plugin()

    @classmethod
    def tearDownClass(cls):
        cls.plugin.closeAllSessions()

    def testCalcAlgsEV(self):
        nItems = 100
        nFeatures = 5
        nRays = 64
        nAlgs = 16

        x = (np.random.rand(nItems, nFeatures) - 0.5).astype(np.float32)
        target = np.random.randint(0, 2, nItems)
        r = (np.random.rand(nRays, nFeatures) - 0.5).astype(np.float32)
        w0 = (np.random.rand(nFeatures, 1) - 0.5).astype(np.float32)

        with self.plugin.createSession(x, target, r) as session:
            rwResult = session.runRandomWalking(w0, nAlgs)
            calcAlgsResult = session.calcAlgs(rwResult.W)
            calcAlgsResult2 = self.plugin.advanced.calcAlgsEV(calcAlgsResult.EV, target)
            self.assertEqual(calcAlgsResult.EC.shape[0], calcAlgsResult2.EC.shape[0])
            for i in range(calcAlgsResult.EC.shape[0]):
                self.assertEqual(calcAlgsResult.EC[i], calcAlgsResult2.EC[i])

            connectivity = self.plugin.advanced.calcAlgsConnectivity(calcAlgsResult.hashes, calcAlgsResult.EC, nItems)
            self.assertEqual(connectivity.upperCon.shape[0], connectivity.lowerCon.shape[0])

if __name__ == '__main__':
    unittest.main()
