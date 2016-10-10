from numpy import *
from loadData import csv2matrix
import operator

import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) 
	labels = ['A','A','B','B']
	return group, labels

#A simple classifier
def classify0(inX, dataSet, labels, k):
	"""
	inX: test input, should be 1*N vector
	dataSet: training data, should be M*N matrix
	labels: training result, should be M*1 vector
	k: return top K nearest result
	So, we have M groups of training data, each has N features
	"""
	dataSetSize = dataSet.shape[0]

	#Calculate the distance
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5

	#Find the k nearest point
	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

	#Return the result by counting the votes
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

#A sample of preparing the data
def file2matrix(filename, featureNum):
	fr = open(filename)
	arrayOfLines = fr.readlines()
	numOfLines = len(arrayOfLines)
	
	returnMat = zeros((numOfLines, featureNum))
	classLabelVet = []
	idx = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split(' ')
		returnMat[idx,:] = listFromLine[0:featureNum]
		classLabelVet.append(str(listFromLine[-1]))
		idx += 1
	return returnMat, classLabelVet


if __name__ == '__main__':
	#g,l = createDataSet()
	#print g,l
	#print classify0([1,1],g,l,3)
	########################
	#Test with dating data set
	'''
	xTrain,lTrain = file2matrix("datingTrainSet.txt", 3)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xTrain[:,1], xTrain[:,2])
	plt.show()

	xTest, lTest = file2matrix("datingTestSet.txt", 3)
	for i in range(3):
		if classify0(xTest[i], xTrain, lTrain, 2)==lTest[i]:
			print "Yes"
		else:
			print "No"
	'''
	########################
	#Test with hand write 4 & 9, get accuracy of 97%
	relative_path = "dataset/"
	xTrain, lTrain = csv2matrix(relative_path+'Xtrain.csv', relative_path+'ytrain.csv')
	xTest, lTest = csv2matrix(relative_path+'Xtest.csv', relative_path+'ytest.csv')
	total = len(lTest)
	cnt = 0
	for i in range(total):
		if classify0(xTest[i], xTrain, lTrain, 7)==lTest[i]:
			cnt += 1
	print cnt, total, float(1.0*cnt/total)








