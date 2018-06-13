'''
@author: John Henry Dahlberg

2018-06-12
'''

from numpy import *
import matplotlib.pyplot as plt
import pickle
import sklearn.preprocessing
from matplotlib.offsetbox import AnchoredText
from sklearn import preprocessing
from copy import *


class Perceptron(object):
    def __init__(self, attributes=None):

        if not attributes:
            raise Exception('Dictionary argument "attributes" is required.')

        self.__dict__ = attributes

        self.d = self.X[0].shape[0]
        self.K = self.targets[0].shape[0]

        # nData = 100 #X[0].shape[1]
        self.trainX = self.X[0]  # [:, :nData]
        self.validX = self.X[1]  # [:, :nData]
        self.testX = self.X[2]  # [:, :nData]
        self.trainTargets = self.targets[0]  # [:, :nData]
        self.validTargets = self.targets[1]  # [:, :nData]
        self.testTargets = self.targets[2]  # [:, :nData]

        self.trainTargetsScalars = self.targetsScalars[0]  # [:nData]
        self.validTargetsScalars = self.targetsScalars[1]  # [:nData]
        self.testTargetsScalars = self.targetsScalars[2]  # [:nData]

        self.normalizeData()

        self.W = []
        self.b = []
        self.layerNeuronSizes = copy(self.hiddenLayerSizes)
        self.layerNeuronSizes.insert(0, self.d)
        self.layerNeuronSizes.append(self.K)
        self.nLayers = len(self.layerNeuronSizes) - 1

        for i in range(0, self.nLayers):
            if self.weightInit == 'He':
                initSigma = sqrt(2 / self.layerNeuronSizes[i])
            else:
                initSigma = 0.001
            self.W.append(initSigma * random.randn(self.layerNeuronSizes[i + 1], self.layerNeuronSizes[i]))
            self.b.append(zeros((self.layerNeuronSizes[i + 1], 1)))

    def normalizeData(self):
        scaler = preprocessing.StandardScaler()

        scaler.fit(self.trainX.T)
        self.trainX = scaler.transform(self.trainX.T).T
        self.validX = scaler.transform(self.validX.T).T
        self.testX = scaler.transform(self.testX.T).T

    def evaluateClassifier(self, X, W=array([[]]), b=array([[]])):
        if not W[0].any():
            W = self.W
            b = self.b

        activations = [X]
        weightedSumsBN = []
        weightedSumsN = []
        meansWS = []
        variancesWS = []
        Vs = []

        finalLayer = self.nLayers
        for layer in range(0, finalLayer):
            weightedSum = dot(W[layer], activations[layer]) + b[layer]

            meanWS = array([mean(weightedSum, axis=1)]).T
            varianceWS = var(weightedSum, axis=1)

            if hasattr(self, 'meansWS') and hasattr(self, 'variancesWS'):
                meanWS = self.alpha * meanWS + (1 - self.alpha) * self.meansWS[layer]
                varianceWS = self.alpha * varianceWS + (1 - self.alpha) * self.variancesWS[layer]
            else:
                meansWS.append(meanWS)
                variancesWS.append(varianceWS)

            weightedSumBN, weightedSumN, V = self.batchNormalize(weightedSum, meanWS, varianceWS)

            weightedSumsN.append(weightedSumN)
            Vs.append(V)
            activation = maximum(0, weightedSumBN)

            weightedSumsBN.append(weightedSumBN)
            if layer < finalLayer - 1:
                activations.append(activation)

        if not hasattr(self, 'meansWS') and not hasattr(self, 'variancesWS'):
            self.meansWS = meansWS
            self.variancesWS = variancesWS

        p = self.softmax(weightedSum)

        return p, activations, weightedSumsBN, weightedSumsN, Vs

    def batchNormalize(self, weightedSum, meanWS, varianceWS):
        eps = 1e-20

        weightedSumN = weightedSum - meanWS
        V = array([varianceWS + eps])

        if self.batchNormalization:
            VinvSqrt = V**-0.5 # <=> sqrt(1/V)
            weightedSumBN = multiply(weightedSumN, VinvSqrt.T)
            return weightedSumBN, weightedSumN, V
        else:
            return deepcopy(weightedSum), weightedSumN, V

    def batchNormBackProp(self, dJdsBN, weightedSumN, V):
        N = dJdsBN.shape[0]

        VinvCbrt = V**-1.5
        dJdv = -0.5 * sum(multiply(multiply(dJdsBN, VinvCbrt), weightedSumN.T), axis=0)

        VinvSqrt = V**-0.5 # <=> sqrt(1/V)
        dJdm = -sum(multiply(dJdsBN, VinvSqrt), axis=0)

        dJds = multiply(dJdsBN, VinvSqrt) + 2 / N * multiply(dJdv, weightedSumN.T) + dJdm / N

        return dJds

    def softmax(self, s):
        p = zeros(s.shape)

        for i in range(s.shape[0]):
            p[i, :] = exp(s[i, :])

        p /= tile(sum(p, axis=0), (s.shape[0], 1))

        return p

    def computeCost(self, X, targets, W=array([[]]), b=array([[]])):
        if not W[0].any():
            W = self.W
            p, activations, weightedSumsBN, weightedSumsN, Vs = self.evaluateClassifier(X)
        else:
            p, activations, weightedSumsBN, weightedSumsN, Vs = self.evaluateClassifier(X, W, b)

        N = X.shape[1]
        #   loss = crossEntropy + regularizationCost
        loss = -(1 / N) * sum(log(diag(dot(targets.T, p)))) + self.lmbda * (sum(W[0] ** 2) + sum(W[1] ** 2))

        return loss

    def computeAccuracy(self, X, targetsScalars):
        p, activations, weightedSumsBN, weightedSumsN, Vs = self.evaluateClassifier(X)
        k = argmax(p, axis=0)

        acc = sum(abs(k - targetsScalars) == 0) / len(k)

        return acc

    def computeGradients(self, X, targets, p, activations, weightedSumsBN, weightedSumsN, Vs):
        N = X.shape[1]
        g = -(targets - p).T

        dJdW = []
        dJdB = []

        finalLayer = self.nLayers

        for layer in range(finalLayer - 1, -1, -1):
            if self.batchNormalization and layer < finalLayer - 1:
                g = self.batchNormBackProp(g, weightedSumsN[layer], Vs[layer])

            dJdWi = dot(g.T, activations[layer].T) / N + 2 * self.lmbda * self.W[layer]
            dJdBi = array([sum(g, axis=0) / N]).T

            if layer > 0:
                g = dot(g, self.W[layer])
                ind = where(weightedSumsBN[layer - 1] > 0, 1, 0)
                g = multiply(g.T, ind).T

            dJdW.append(dJdWi)
            dJdB.append(dJdBi)

        dJdW.reverse()
        dJdB.reverse()

        return dJdW, dJdB

    def miniBatchGD(self):
        N = self.trainX.shape[1]

        trainX = self.trainX
        validX = self.validX
        trainTargets = self.trainTargets
        validTargets = self.validTargets

        validTargetsScalars = self.validTargetsScalars

        updatedW = []
        updatedB = []

        for layer in range(0, self.nLayers):
            updatedW.append(self.W[layer])
            updatedB.append(self.b[layer])

        trainLosses = []
        validLosses = []
        epochs = []

        if self.plotProcess:
            plt.close("all")
            fig = plt.figure()

        nBatches = int(N / self.nBatch)
        constants = 'Max Epochs = ' + str(self.nEpochs) + '\n# Batches = ' + str(nBatches) \
                    + '\n# Hidden neurons = ' + str(self.hiddenLayerSizes) \
                    + '\n' + r'$\eta$ = ' + "{:.2e}".format(self.eta) + '\n' \
                    + r'$\eta_{Decay}$ = ' + str(self.etaDecayFactor) + '\n' \
                    + r'$\lambda$ = ' + "{:.2e}".format(self.lmbda) + '\n' \
                    + r'$\rho$ = ' + str(self.rho) + '\n' \
                    + r'$\alpha$ = ' + str(self.alpha) + '\n' \
                    + 'Weight initialization: ' + str(self.weightInit) + '\n' \
                    + 'Batch Normalization: ' + str(self.batchNormalization) + '\n' \
                    + 'Early Stopping: ' + str(self.earlyStopping) + \
                    '\n' + '# Samples for Train, Valid and Test:' + \
                    '\n' + "{:.2e}".format(self.trainX.shape[1]) + ', ' + "{:.2e}".format(self.validX.shape[1]) + \
                    ', ' + "{:.2e}".format(self.testX.shape[1])

        earlyStopping = False

        for i in range(1, self.nEpochs + 1):
            vW = []
            vB = []

            for layer in range(0, self.nLayers):
                vW.append(zeros(self.W[layer].shape))
                vB.append(zeros(self.b[layer].shape))

            if not earlyStopping:
                for j in range(nBatches):
                    startIndex = j * self.nBatch
                    endIndex = (j + 1) * self.nBatch
                    trainXbatch = trainX[:, startIndex: endIndex]
                    trainTargetsBatch = trainTargets[:, startIndex: endIndex]

                    p, activations, weightedSumsBN, weightedSumsN, Vs = self.evaluateClassifier(trainXbatch)
                    [grad_W, grad_b] = self.computeGradients(trainXbatch, trainTargetsBatch, p, activations, weightedSumsBN, weightedSumsN, Vs)

                    for layer in range(0, self.nLayers):
                        vW[layer] = self.eta * grad_W[layer] + self.rho * vW[layer]
                        vB[layer] = self.eta * grad_b[layer] + self.rho * vB[layer]

                        updatedW[layer] -= vW[layer]
                        updatedB[layer] -= vB[layer]

                    self.W = updatedW
                    self.B = updatedB

                self.eta *= self.etaDecayFactor

                trainLoss = self.computeCost(trainX, trainTargets)
                validLoss = self.computeCost(validX, validTargets)

                if self.earlyStopping and len(validLosses) > 4 and validLoss > validLosses[-1]:
                    print('Stopping due to increase of validation loss')
                    break

                bestValidAcc = self.computeAccuracy(validX, validTargetsScalars)
                print('Epoch ' + str(i) + '/' + str(self.nEpochs) + ':')
                print('Validation accuracy = ' + str(round(bestValidAcc * 100, 2)) + '%')

                trainLosses.append(trainLoss)
                validLosses.append(validLoss)

                if trainLoss > 3 * trainLosses[0]:
                    print('trainLoss > 3*trainLosses[0]')
                    break

                if self.plotProcess:
                    epochs.append(i)

                    plt.clf()
                    ax = fig.add_subplot(111)
                    fig.subplots_adjust(top=0.85)
                    anchored_text = AnchoredText(constants, loc=3)
                    ax.add_artist(anchored_text)

                    plt.ylabel('Loss')
                    plt.xlabel('Epochs')
                    plt.plot(epochs, trainLosses, label='Training', LineWidth=2)
                    plt.plot(epochs, validLosses, label='Validation', color='orange', LineWidth=2)
                    plt.legend(loc='upper right')
                    plt.grid()
                    plt.pause(0.001)

        acc = self.computeAccuracy(self.testX, self.testTargetsScalars)
        print('\nFinal test accuracy = ' + str(round(acc * 100, 2)) + '%')

        return bestValidAcc

    def computeGradsNumSlow(self, X, targets):

        h = 1e-4
        gradWnum = []
        gradBnum = []

        for i in range(0, self.nLayers):
            gradWnum.append(zeros(self.W[i].shape))
            gradBnum.append(zeros(self.b[i].shape))

        for layer in range(0, self.nLayers):
            for i in range(self.b[layer].shape[0]):
                bTry = deepcopy(self.b)
                bTry[layer][i, 0] -= h
                c1 = self.computeCost(X, targets, self.W, bTry)

                bTry = deepcopy(self.b)
                bTry[layer][i, 0] += h
                c2 = self.computeCost(X, targets, self.W, bTry)
                gradBnum[layer][i, 0] = (c2 - c1) / (2 * h)

            iS = [0, 1, 2, self.W[layer].shape[0] - 3, self.W[layer].shape[0] - 2, self.W[layer].shape[0] - 1]
            jS = [0, 1, 2, self.W[layer].shape[1] - 3, self.W[layer].shape[1] - 2, self.W[layer].shape[1] - 1]
            for i in iS:
                for j in jS: # range(self.W[layer].shape[1]):
                    WTry = deepcopy(self.W)
                    WTry[layer][i, j] -= h
                    c1 = self.computeCost(X, targets, WTry, self.b)

                    WTry = deepcopy(self.W)
                    WTry[layer][i, j] += h
                    c2 = self.computeCost(X, targets, WTry, self.b)
                    gradWnum[layer][i, j] = (c2 - c1) / (2 * h)

                #if i % (self.W[layer].shape[0] / 100) == 0:
                #    print('GradNum process: ' + str(i / self.W[0].shape[0] * 100))

        return gradWnum, gradBnum

    def testComputedGradients(self):
        x = self.trainX[:, :7]
        y = self.trainTargets[:, :7]

        epsilon = 1e-10
        p, activations, weightedSumsBN, weightedSumsN, Vs = self.evaluateClassifier(x)
        [grad_W, grad_b] = self.computeGradients(x, y, p, activations,
                                                 weightedSumsBN, weightedSumsN, Vs)

        differenceWSmall = []
        differenceBSmall = []

        for layer in range(self.nLayers):
            grad_W_Num, grad_b_Num = self.computeGradsNumSlow(x, y)

            differenceW = abs(grad_W[layer] - grad_W_Num[layer]) / maximum(epsilon,
                                                                           (abs(grad_W[layer]) + abs(grad_W_Num[layer])))
            differenceB = abs(grad_b[layer] - grad_b_Num[layer]) / maximum(epsilon,
                                                                           (abs(grad_b[layer]) + abs(grad_b_Num[layer])))

            # Only calculate first and last three rows and columns
            differenceWSmall.append(zeros((7, 7)))
            differenceBSmall.append(zeros((1, 7)))

            iS = [0, 1, 2, self.W[layer].shape[0] - 3, self.W[layer].shape[0] - 2, self.W[layer].shape[0] - 1]
            jS = [0, 1, 2, self.W[layer].shape[1] - 3, self.W[layer].shape[1] - 2, self.W[layer].shape[1] - 1]

            bS = [0, 1, 2, self.b[layer].shape[0] - 3, self.b[layer].shape[0] - 2, self.b[layer].shape[0] - 1]

            for i in range(6):
                for j in range(6):
                    differenceWSmall[layer][i, j] = "{:.2e}".format(differenceW[iS[i], jS[j]])
                differenceBSmall[layer][0, i] = "{:.2e}".format(differenceB[bS[i]][0])

            print('\nLayer ' + str(layer + 1) + ':')
            print('Absolute differences of W:')
            print(differenceWSmall[layer])
            # print(pMatrix(differenceWSmall[layer]))

            print('Absolute differences of B:')
            print(differenceBSmall[layer])
            # print(pMatrix(differenceBSmall[layer]))


def pMatrix(array):
    #   Converts matrixes to LaTeX code
    rows = str(array).replace('[', '').replace(']', '').splitlines()
    rowString = [r'\begin{pmatrix}']
    for row in rows:
        rowString += [r'  \num{' + r'} & \num{'.join(row.split()) + r'}\\']

    rowString += [r'\end{pmatrix}']

    return '\n'.join(rowString)


def unpickle(fileName):
    with open(fileName, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
    return dict


def loadBatch(useFullSet=True):
    fileNames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    path = 'cifar-10-python/cifar-10-batches-py/'

    batchesX = []
    batchesScalarsY = []
    batchesY = []
    X = []
    targetsScalars = []
    targetsOneHot = []
    if not useFullSet:
        fileNames = [fileNames[0], fileNames[1], fileNames[-1]]

    for i in range(len(fileNames)):
        file = fileNames[i]
        dict = unpickle(path + file)
        x = array(dict[b'data'] / 255).T
        y = array(dict[b'labels']).T
        binarizer = sklearn.preprocessing.LabelBinarizer()
        binarizer.fit(range(max(y.astype(int)) + 1))
        Y = array(binarizer.transform(y.astype(int))).T

        batchesX.append(x)
        batchesScalarsY.append(y)
        batchesY.append(Y)

    if len(fileNames) == 3:
        X.append(batchesX[0])
        X.append(batchesX[1])
        X.append(batchesX[2])

        targetsScalars.append(batchesScalarsY[0])
        targetsScalars.append(batchesScalarsY[1])
        targetsScalars.append(batchesScalarsY[2])

        targetsOneHot.append(batchesY[0])
        targetsOneHot.append(batchesY[1])
        targetsOneHot.append(batchesY[2])
    else:
        X.append(concatenate((batchesX[0], batchesX[1], batchesX[2], batchesX[3], batchesX[4][:, :9000]), axis=1))
        X.append(batchesX[4][:, 9000:])
        X.append(batchesX[5])

        targetsScalars.append(concatenate((batchesScalarsY[0], batchesScalarsY[1], batchesScalarsY[2],
                                           batchesScalarsY[3], batchesScalarsY[4][:9000])))
        targetsScalars.append(batchesScalarsY[4][9000:])
        targetsScalars.append(batchesScalarsY[5])

        targetsOneHot.append(
            concatenate((batchesY[0], batchesY[1], batchesY[2], batchesY[3], batchesY[4][:, :9000]), axis=1))
        targetsOneHot.append(batchesY[4][:, 9000:])
        targetsOneHot.append(batchesY[5])

    return X, targetsOneHot, targetsScalars


def generateRandomPair(attributes, nPairs, fileName):
    bestPairs = []

    for i in range(nPairs):
        print('\nPairing ', i)

        eMin = -2
        eMax = -2
        ex = eMin + (eMax - eMin)*random.random()
        bMin = 5
        bMax = 7
        b = bMin + (bMax - bMin)*random.random()
        learningrate = b*10**ex
        print('Random learningrate = ' + "{:.3g}".format(learningrate))

        eMin = -7
        eMax = -6
        ex = eMin + (eMax - eMin)*random.uniform(0, 1)
        bMin = 1
        bMax = 5
        b = bMin + (bMax - bMin)*random.random()
        regularization = b*10**ex
        print('Random regularization = ' + "{:.3g}".format(regularization))
        attributes['eta'] = learningrate
        attributes['lmbda'] = regularization
        attributes['nEpochs'] = 5
        attributes['plotProcess'] = False

        perceptron = Perceptron(attributes)
        bestValidAcc = perceptron.miniBatchGD()

        bestPairs.append((learningrate, regularization, bestValidAcc))

    savetxt(fileName, bestPairs, delimiter=',')
    print(bestPairs)


def getTopTuples(fileName, nPairs):
    # Load 3 best pairs
    bestPairs = loadtxt(fileName, delimiter=',')
    orderedParings = sorted(bestPairs, key=lambda x: x[2])[nPairs-3:]
    print(orderedParings)
    savetxt(fileName+'Top', orderedParings, delimiter=',')


def main():

    useFullSet = False
    generateRandomEtaLambdaPair = False
    testComputedGradients = True

    # Data, One hot targets, Targets with scalar label
    X, targets, targetsScalars = loadBatch(useFullSet)

    attributes = {
        'X': X,
        'targets': targets,
        'targetsScalars': targetsScalars,
        'eta': 6.28e-2,
        'etaDecayFactor': 0.95,
        'lmbda': 3.16e-6,
        'rho': 0.9,
        'alpha': 0.99,
        'nEpochs': 10,
        'nBatch': 100,
        'hiddenLayerSizes': [50, 30],
        'weightInit': 'He',
        'batchNormalization': True,
        'earlyStopping': True,
        'plotProcess': True
    }

    if generateRandomEtaLambdaPair:
        nPairs = 25
        fileName = 'bestPairsFineMLP4'
        generateRandomPair(attributes, nPairs, fileName)
        getTopTuples(fileName, nPairs)
    else:
        perceptron = Perceptron(attributes)
        bestValidAcc = perceptron.miniBatchGD()

    if testComputedGradients:
        perceptron.testComputedGradients()


if __name__ == '__main__':
    random.seed(0)
    main()
    plt.show()
