import collections
import ctypes
import numpy as np
import os

__author__ = 'Alexander Frey'


# Python wrapper around LinearSampling.dll
class Plugin:
    def __init__(self):
        self.dll = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__)) + '\\LinearSamplingCPU.dll')
        self.c_float_p = ctypes.POINTER(ctypes.c_float)
        self.c_int_p = ctypes.POINTER(ctypes.c_int)
        self.c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
        self.c_uint_p = ctypes.POINTER(ctypes.c_uint)
        self.advanced = PluginAdvanced(self)

    def verifyCall(self, retval):
        if retval < 0:
            raise Exception("Some failure happened, check LinearSampling.log for more details.")

    def closeAllSessions(self):
        self.verifyCall(self.dll.closeAllSessions())

    def createSession(self, X, target, R=None):
        nItems = X.shape[0]
        nFeatures = X.shape[1]

        if R is None:
            nRays = 4 * nFeatures
            R = (np.random.rand(nRays, nFeatures) - 0.5).astype(np.float32)
        else:
            nRays = R.shape[0]
            if R.shape[1] != nFeatures:
                raise Exception("createSession() failed, X.shape[1] != R.shape[1]")

        X = np.asfortranarray(X)
        target = np.asfortranarray(target)
        R = np.asfortranarray(R)

        x_p = X.ctypes.data_as(self.c_float_p)
        r_p = R.ctypes.data_as(self.c_float_p)
        target_p = target.ctypes.data_as(self.c_int_p)
        sessionId = -1  # ask the DLL to auto-assign sessionId.
        deviceId = 0    # deviceId is used to identify GPU. For CPU version of the dll this is unused.
        sessionId = self.dll.createSession(x_p, target_p, r_p, nItems, nFeatures, nRays, deviceId, sessionId)
        return Session(self, sessionId)

    def replayRecording(self, filename):
        filename_p = ctypes.create_string_buffer(filename)
        self.verifyCall(self.dll.replayRecording(filename_p))

    def setLogLevel(self, logLevel):
        self.verifyCall(self.dll.setLogLevel(logLevel))

    def startRecording(self, filename):
        filename_p = ctypes.create_string_buffer(filename)
        self.verifyCall(self.dll.startRecording(filename_p))

    def stopRecording(self):
        self.verifyCall(self.dll.stopRecording())

    def performCrossValidation(self, ev, ec, nIters, nTrainItems=None, randomSeed=0):
        ev = np.asfortranarray(ev)
        ec = np.asfortranarray(ec)
        ev_p = ev.ctypes.data_as(self.c_uint8_p)
        ec_p = ec.ctypes.data_as(self.c_int_p)

        nItems = ev.shape[0]
        nAlgs = ev.shape[1]
        if ec.shape[0] != nAlgs:
            raise Exception("performCrossValidation() failed, ev.shape[1] != ec.shape[0]")

        if nTrainItems is None:
            nTrainItems = nItems / 2

        trainEC = np.asfortranarray(np.zeros((nIters, 1)).astype(np.int))
        testEC = np.asfortranarray(np.zeros((nIters, 1)).astype(np.int))
        self.verifyCall(
            self.dll.performCrossValidation(
                ev_p, ec_p, nItems, nTrainItems, nAlgs, nIters, randomSeed, trainEC.ctypes.data_as(self.c_int_p),
                testEC.ctypes.data_as(self.c_int_p)))

        predictedBias = np.mean(testEC) / (nItems - nTrainItems) - np.mean(trainEC) / nTrainItems
        CrossValidationResult = collections.namedtuple('CrossValidationResult', 'trainEC testEC predictedBias')
        return CrossValidationResult(trainEC, testEC, predictedBias)


# Wrapper around advanced methods in LinearSampling.dll
class PluginAdvanced:
    def __init__(self, plugin):
        self.plugin = plugin

    def calcAlgsEV(self, EV, target):
        if EV.shape[0] != target.shape[0]:
            raise Exception("EV.shape[0] != target.shape[0]")

        nItems = EV.shape[0]
        nAlgs = EV.shape[1]
        EC = np.asfortranarray(np.zeros((nAlgs, 1)).astype(np.int))
        hashes = np.asfortranarray(np.zeros((nAlgs, 1)).astype(np.uint))
        self.plugin.verifyCall(self.plugin.dll.calcAlgsEV(
            EV.ctypes.data_as(self.plugin.c_uint8_p),
            target.ctypes.data_as(self.plugin.c_int_p),
            nItems,
            nAlgs,
            EC.ctypes.data_as(self.plugin.c_int_p),
            hashes.ctypes.data_as(self.plugin.c_uint_p)))

        CalcAlgsEVResult = collections.namedtuple('CalcAlgsEVResult', 'EC hashes')
        return CalcAlgsEVResult(EC, hashes)

    def calcAlgsConnectivity(self, hashes, EC, nItems):
        if hashes.shape[0] != EC.shape[0]:
            raise Exception("hashes.shape[0] != EC.shape[0]")

        nAlgs = hashes.shape[0]
        upperCon = np.asfortranarray(np.zeros((nAlgs, 1)).astype(np.int))
        lowerCon = np.asfortranarray(np.zeros((nAlgs, 1)).astype(np.int))

        self.plugin.verifyCall(self.plugin.dll.calcAlgsConnectivity(
            hashes.ctypes.data_as(self.plugin.c_uint_p),
            EC.ctypes.data_as(self.plugin.c_int_p),
            nAlgs,
            nItems,
            upperCon.ctypes.data_as(self.plugin.c_int_p),
            lowerCon.ctypes.data_as(self.plugin.c_int_p)))

        CalcAlgsConnectivityResult = collections.namedtuple('CalcAlgsConnectivityResult', 'upperCon lowerCon')
        return CalcAlgsConnectivityResult(upperCon, lowerCon)


# Represents an active session in LinearSampling.dll
# This class manages native resources and must be used under "with" statement.
# Internally this class keeps the sessionId and makes sure that it is disposed.
class Session:
    def __init__(self, lsPlugin, sessionId):
        self.lsPlugin = lsPlugin
        self.sessionId = sessionId

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.sessionId != -1:
            self.close()

    def calcAlgs(self, W):
        sessionStats = self.getStats()
        if sessionStats.nFeatures != W.shape[1]:
            raise Exception("sessionStats.nFeatures != W.share[1]")

        nAlgs = W.shape[0]
        scores = np.asfortranarray(np.zeros((sessionStats.nItems, nAlgs)).astype(np.float32))
        EV = np.asfortranarray(np.zeros((sessionStats.nItems, nAlgs)).astype(np.uint8))
        EC = np.asfortranarray(np.zeros((nAlgs, 1)).astype(np.int))
        hashes = np.asfortranarray(np.zeros((nAlgs, 1)).astype(np.uint))
        self.lsPlugin.verifyCall(self.lsPlugin.dll.calcAlgs(
            self.sessionId,
            W.ctypes.data_as(self.lsPlugin.c_float_p),
            nAlgs,
            scores.ctypes.data_as(self.lsPlugin.c_float_p),
            EV.ctypes.data_as(self.lsPlugin.c_uint8_p),
            EC.ctypes.data_as(self.lsPlugin.c_int_p),
            hashes.ctypes.data_as(self.lsPlugin.c_uint_p)))

        CalcAlgsResult = collections.namedtuple('CalcAlgsResult', 'scores, EV, EC, hashes')
        return CalcAlgsResult(scores, EV, EC, hashes)

    def calcQEpsCombinatorial(self, W, isSource, epsValues, nTrainItems=-1, boundType=0):
        sessionStats = self.getStats()
        if W.shape[1] != sessionStats.nFeatures:
            raise Exception("W.shape[1] != sessionStats.nFeatures")

        if nTrainItems < 0:
            nTrainItems = sessionStats.nItems / 2

        if nTrainItems <= 0 or nTrainItems >= sessionStats.nItems:
            raise Exception("nTrainItems <= 0 or nTrainItems >= sessionStats.nItems")
        if isSource.shape[0] != W.shape[0]:
            raise Exception("isSource.shape[0] != W.shape[0]")

        nAlgs = W.shape[0]
        nEpsValues = epsValues.shape[0]

        QEps = np.asfortranarray(np.zeros((nAlgs, nEpsValues)).astype(np.float32))

        self.lsPlugin.verifyCall(self.lsPlugin.dll.calcQEpsCombinatorial(
            self.sessionId,
            W.ctypes.data_as(self.lsPlugin.c_float_p),
            isSource.ctypes.data_as(self.lsPlugin.c_uint8_p),
            epsValues.ctypes.data_as(self.lsPlugin.c_float_p),
            nTrainItems,
            nAlgs,
            nEpsValues,
            boundType,
            QEps.ctypes.data_as(self.lsPlugin.c_float_p)))

        QEps = QEps.sum(axis=0)
        return QEps

    def close(self):
        self.lsPlugin.verifyCall(self.lsPlugin.dll.closeSession(self.sessionId))
        self.sessionId = -1

    def getStats(self):
        nItems = ctypes.c_int()
        nFeatures = ctypes.c_int()
        nRays = ctypes.c_int()
        deviceId = ctypes.c_int()
        self.lsPlugin.verifyCall(
            self.lsPlugin.dll.getSessionStats(self.sessionId, ctypes.byref(nItems), ctypes.byref(nFeatures),
                                              ctypes.byref(nRays), ctypes.byref(deviceId)))
        SessionStats = collections.namedtuple('SessionStats', 'nItems nFeatures nRays deviceId')
        return SessionStats(nItems.value, nFeatures.value, nRays.value, deviceId.value)

    def getXR(self):
        sessionStats = self.getStats()
        XR = np.asfortranarray(np.zeros((sessionStats.nItems, sessionStats.nRays)).astype(np.float32))
        XR_p = XR.ctypes.data_as(self.lsPlugin.c_float_p)
        self.lsPlugin.dll.getXR(self.sessionId, XR_p)
        return XR

    def runRandomWalking(self, w, nAlgs, nWalks = 1, nIters=-1, nErrorsLimit=-1, allowSimilar=False, pTransition=0.8,
                         randomSeed=0):
        RunRandomWalkingResult = collections.namedtuple('RunRandomWalkingResult', 'W isSource')
        nFeatures = w.shape[0]
        w0 = np.tile(w, (nWalks, 1))
        sessionStats = self.getStats()

        if sessionStats.nFeatures != nFeatures:
            raise Exception('sessionStats.nFeatures != w0.shape[1]')

        W = np.asfortranarray(np.zeros((nAlgs, nFeatures)).astype(np.float32))
        isSource = np.asfortranarray(np.zeros((nAlgs, 1)).astype(np.uint8))

        w0_p = w0.ctypes.data_as(self.lsPlugin.c_float_p)
        W_p = W.ctypes.data_as(self.lsPlugin.c_float_p)
        isSource_p = isSource.ctypes.data_as(self.lsPlugin.c_uint8_p)

        pTransition_p = (ctypes.c_float * 1)()
        pTransition_p[0] = pTransition;
        #pTransition.ctypes.data_as(self.c_float_p)
        nAlgs = self.lsPlugin.dll.runRandomWalking(self.sessionId, w0_p, nWalks, nAlgs, nIters, nErrorsLimit,
                                                   allowSimilar, pTransition_p, randomSeed, W_p, isSource_p)
        self.lsPlugin.verifyCall(nAlgs)

        return RunRandomWalkingResult(W, isSource)