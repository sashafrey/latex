import LinearSampling.dataset as lsData

__author__ = 'Administrator'

import unittest


class testDataset(unittest.TestCase):
    def testLoad(self):
        filename = 'C:\\Storage\\vft11ccas\\Source\\DataUCI\\pima\\pima-indians-diabetes.data'
        dataset = lsData.loadFromFile(filename)
        self.assertTrue(dataset.nItems >= 2)
        self.assertTrue(dataset.nFeatures > 0)
        trainSampleSize = dataset.nItems / 3
        samples = dataset.split(trainSampleSize)
        self.assertEqual(samples.TrainSample.nItems, trainSampleSize)
        self.assertEqual(samples.TestSample.nItems, dataset.nItems - trainSampleSize)

if __name__ == '__main__':
    unittest.main()
