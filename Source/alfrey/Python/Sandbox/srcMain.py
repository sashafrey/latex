__author__ = 'Alexander Frey'

# Import modules
import numpy as np
import statsmodels.api as sm
import LinearSampling.api as lsApi
import LinearSampling.dataset as lsData
np.random.mtrand.seed(10)

# Load dataset (Pima from UDI repository; 768 items, 8 features)
dataset = lsData.loadFromFile('pima-indians-diabetes.data')
dataset.X = sm.add_constant(dataset.X, prepend=False)

# Split dataset into train and test samples
samples = dataset.split(dataset.nItems / 2)
trainSample = samples.TrainSample
testSample = samples.TestSample

# Tune logistic regression and show the model
logisticRegression = sm.GLM(trainSample.target, trainSample.X, family=sm.families.Binomial())
model = logisticRegression.fit().params
print "Model : [ " + ", ".join(format(x, ".3f") for x in model) + " ]"

# Check error rate on train and test sample
trainPredictions = logisticRegression.predict(model, trainSample.X)
trainErrorRate = float(sum((trainPredictions > 0.5) != trainSample.target)) / trainSample.nItems

testPredictions = logisticRegression.predict(model, testSample.X)
testErrorRate = float(sum((testPredictions > 0.5) != testSample.target)) / testSample.nItems

print "Train error rate : " + '%.3f' % (100 * trainErrorRate) + " %"
print "Test error rate  : " + '%.3f' % (100 * testErrorRate) + " %"
print "True bias        : " + '%.3f' % (100 * (testErrorRate - trainErrorRate)) + " %"

# Erase test sample
testSample = None

# Estimate bias from train sample
plugin = lsApi.Plugin()
#plugin.startRecording("C:\\recording.dat")
with plugin.createSession(trainSample.X, trainSample.target) as session:
    modelNeighbours = session.runRandomWalking(model, 256, 8192, allowSimilar=False).W
    modelNeighboursStats = session.calcAlgs(modelNeighbours)
    predictedBias = plugin.performCrossValidation(modelNeighboursStats.EV, modelNeighboursStats.EC, 1024).predictedBias
#plugin.stopRecording()
print "Predicted bias   : " + '%.3f' % (100 * predictedBias) + " %"
