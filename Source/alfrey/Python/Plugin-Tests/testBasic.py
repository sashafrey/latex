import numpy as np
import os.path
import LinearSampling.api as lsApi
import unittest

__author__ = 'Alexander Frey'


class testBasic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = lsApi.Plugin()

    @classmethod
    def tearDownClass(cls):
        cls.plugin.closeAllSessions()

    def testCanary(self):
        self.assertTrue(True)

    def testCreateSession(self):
        nItems = 12
        nFeatures = 4
        x = (np.random.rand(nItems, nFeatures) - 0.5).astype(np.float32)
        target = np.random.randint(0, 2, nItems)

        with self.plugin.createSession(x, target) as session:
            sessionStats = session.getStats()
            self.assertEqual(sessionStats.nFeatures, nFeatures)
            self.assertEqual(sessionStats.nItems, nItems)

    def testStartStopRecording(self):
        filename = "temp.recording"
        if os.path.isfile(filename):
            os.remove(filename)

        self.assertFalse(os.path.isfile(filename))
        self.plugin.startRecording(filename)
        self.plugin.stopRecording()
        self.assertTrue(os.path.isfile(filename))
        self.plugin.replayRecording(filename)
        os.remove(filename)

    def testSetLogLevel(self):
        self.plugin.setLogLevel(1024)
        self.plugin.setLogLevel(0)

    def testRunRandomWalking(self):
        nItems = 120
        nFeatures = 8
        nAlgs = 128
        nWalks = 1
        nEpsValues = 100
        nCVIters = 100

        x = (np.random.rand(nItems, nFeatures) - 0.5).astype(np.float32)
        target = np.random.randint(0, 2, nItems)
        w0 = (np.random.rand(nFeatures, 1) - 0.5).astype(np.float32)

        with self.plugin.createSession(x, target) as session:
            rwResult = session.runRandomWalking(w0, nWalks, nAlgs, allowSimilar=True)
            calcAlgsResult = session.calcAlgs(rwResult.W)
            for i in range(calcAlgsResult.EC.shape[0] - 1):
                self.assertTrue(abs(calcAlgsResult.EC[i] - calcAlgsResult.EC[i + 1]) == 1)

            epsValues = np.arange(nEpsValues).astype(np.float32) / nEpsValues
            QEps = session.calcQEpsCombinatorial(rwResult.W, rwResult.isSource, epsValues)
            for i in range(QEps.shape[0] - 1):
                self.assertTrue(QEps[i] >= QEps[i + 1])

        cvResult = self.plugin.performCrossValidation(calcAlgsResult.EV, calcAlgsResult.EC, nCVIters)
        self.assertTrue(np.all(cvResult.trainEC >= 0) and np.all(cvResult.trainEC <= nItems))
        self.assertTrue(np.all(cvResult.testEC >= 0) and np.all(cvResult.testEC <= nItems))
        self.assertTrue(np.any(cvResult.trainEC > 0) and np.any(cvResult.trainEC < nItems))
        self.assertTrue(np.any(cvResult.testEC > 0) and np.any(cvResult.testEC < nItems))

if __name__ == '__main__':
    unittest.main()
