import numpy as np

#Read the data from csv
def csv2matrix(xFilename, yFilename):
	#Split the csv file with ","
	returnMat = np.genfromtxt(xFilename, delimiter=',')
	classLabelVet = np.genfromtxt(yFilename, delimiter=',')
	return returnMat, classLabelVet