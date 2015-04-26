__author__ = 'eslamelsawy'

import matplotlib.pyplot as pyplot
import numpy as numpy
import pylab

# DataSet #1
allTrainData = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample1_train.txt")
testdata = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample1_test.txt")

# DataSet #2
# allTrainData = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample2_train.txt")
# testdata = numpy.loadtxt("/Users/eslamelsawy/Desktop/ML Course/Course 2/Hw #1/HW1_sample_data/hw1_sample2_test.txt")

def cubicSpline(x,r):
    return pow(max(0, x-r),3)

def getFirstBasis():
    def myFunction(x):
        return 1
    return myFunction

def getSecondBasis():
    def myFunction(x):
        return x
    return myFunction

def getCubicBasis(r1,r2,r3):
    def myFunction(x):
        return ((cubicSpline(x,r1)-cubicSpline(x,r2))/((r2-r1)*1.0)) - ((cubicSpline(x,r2)-cubicSpline(x,r3))/((r3-r2)*1.0))
    return myFunction

def calculateMSE(expectedY, predictedY):
    MSE = 0.0;
    for i in range(0,len(expectedY)):
        MSE+= pow(expectedY[i]-predictedY[i],2)
    return MSE/ (len(expectedY)*1.0)

def Main():
    foldSize = len(allTrainData)/5
    trainsetMSEList = []
    validationsetMSEList= []

    bestBasisFunctions = []
    bestNumberOfKnots = -1;
    bestMSE = 9223372036854775807;

    for numberOfKnots in range(2,9):
        # Knots
        knots = numpy.arange(-4,4,8/(numberOfKnots-1.0))
        knots = numpy.append(knots,4)
        print ("Number of Knots = %d"%numberOfKnots);
        print ("Knots are : %s"%knots)

        # Basis Functions
        basisFunctions = []
        basisFunctions.append(getFirstBasis())
        basisFunctions.append(getSecondBasis())
        for i in range(0, len(knots)-2):
            basisFunctions.append(getCubicBasis(knots[i], knots[i+1], knots[i+2]))

        trainsetMSESum = 0.0
        validationsetMSESum = 0.0

        # divide training set into 5 folds
        for validationsetIndex in range(0,5):

            # Divide All Training Data into training and validation
            validationData = allTrainData.__getslice__(foldSize*validationsetIndex, foldSize*(validationsetIndex+1))
            trainData = allTrainData.tolist()
            trainData.__delslice__(foldSize*validationsetIndex, foldSize*(validationsetIndex+1))

            # Basis Vector
            X = [a[0] for a in trainData]
            Y = [a[1] for a in trainData]
            basisVector = []
            for x in X:
                dataPoint = []
                for i in range(0,len(basisFunctions)):
                    dataPoint.append(basisFunctions[i](x))
                basisVector.append(dataPoint)

            # Train Model
            w, residuals, _, _ = numpy.linalg.lstsq(basisVector, Y)
            trainsetMSESum += residuals[0]/(len(X)*1.0)

            # Validation Set
            Xvalidation = [a[0] for a in validationData]
            expectedY = [a[1] for a in validationData]
            predictedY = []
            for x in Xvalidation:
                y = 0
                for i in range(0,len(basisFunctions)):
                    y += (w[i]*basisFunctions[i](x))
                predictedY.append(y)
            validationsetMSESum += calculateMSE (expectedY, predictedY)

        averageTrainsetMSE = (trainsetMSESum/5.0)
        trainsetMSEList.append(averageTrainsetMSE)
        print("Average training data MSE = %f"%averageTrainsetMSE)

        averageValidationsetMSE = (validationsetMSESum/5.0)
        validationsetMSEList.append(averageValidationsetMSE)
        print("Average validation data MSE = %f"%averageValidationsetMSE)

        if(averageValidationsetMSE< bestMSE):
            bestMSE = averageValidationsetMSE
            bestBasisFunctions = basisFunctions
            bestNumberOfKnots = numberOfKnots
        print ("========================")


    # plotting
    print("TrainingsetMSEvsKnots = %s"%trainsetMSEList)
    print("ValidationsetMSEvsKnots = %s"%validationsetMSEList)
    print("Best K = %d"%bestNumberOfKnots)
    knots = numpy.arange(2,9)
    trainingPlot = pyplot.plot(knots, trainsetMSEList, "ro-",markersize=8)
    validationPlot = pyplot.plot(knots, validationsetMSEList, "bs-",markersize=8)
    pyplot.legend( ["Training MSE", "Validation MSE"])
    pyplot.title("MSE VS Number of Knots, Least MSE at K = %d"%bestNumberOfKnots)
    pyplot.show()

    # Train Model with the whole training data with the best number of knots
    X = [a[0] for a in allTrainData]
    Y = [a[1] for a in allTrainData]
    basisVector = []
    for x in X:
        dataPoint = []
        for i in range(0,len(bestBasisFunctions)):
            dataPoint.append(bestBasisFunctions[i](x))
        basisVector.append(dataPoint)
    w, _, _, _ = numpy.linalg.lstsq(basisVector, Y)

    # Test Set
    Xtest= [a[0] for a in testdata]
    expectedY = [a[1] for a in testdata]
    predictedY = []
    for x in Xtest:
        y = 0
        for i in range(0,len(bestBasisFunctions)):
            y += (w[i]*bestBasisFunctions[i](x))
        predictedY.append(y)
    testSetMSE = calculateMSE (expectedY, predictedY)
    print("Training Data MSE at (k=%d) = %f"%(bestNumberOfKnots,testSetMSE))


Main()

