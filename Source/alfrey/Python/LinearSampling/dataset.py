import collections
import numpy as np

__author__ = 'Alexander Frey'


class Dataset(object):
    def __init__(self, X, target):
        if X.shape[0] != target.shape[0]:
            raise Exception('Can not create dataset from X and target --- nItems mismatch')

        self.X = X
        self.target = target

    def selectItems(self, mask):
        newX = self.X[mask, :]
        newTarget = self.target[mask, :]
        return Dataset(newX, newTarget)

    def split(self, nTrainItems):
        sampleIds = np.random.mtrand.choice(self.nItems, nTrainItems, replace=False)
        mask = np.zeros(self.nItems, dtype=np.bool)
        mask[sampleIds] = 1
        trainSample = self.selectItems(mask)
        testSample = self.selectItems(~mask)
        DatasetSplit = collections.namedtuple('DatasetSplit', 'TrainSample TestSample')
        return DatasetSplit(trainSample, testSample)

    @property
    def nItems(self):
        return self.X.shape[0]

    @property
    def nFeatures(self):
        return self.X.shape[1]


def loadFromFile(filename):
    f = np.genfromtxt(filename, delimiter=',', missing_values='nan')
    nFeatures = f.shape[1] - 1
    target = f[:, nFeatures]
    X = f[:, 0:nFeatures]
    return Dataset(X, target)
