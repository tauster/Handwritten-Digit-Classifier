from utils import load_train_small, load_train, load_valid, load_test
from plot_digits import plot_digits
from l2_distance import l2_distance
import numpy as np
from run_knn import run_knn
import matplotlib.pyplot as plt

"""
CSC 2515 - Assignment 1
Tausif Sharif

Notes:
	- Runs the run_knn.py functions from here
	- Will show and save relevant plots
"""

trainInputs, trainTargets = load_train()
smallInputs, smallTargets = load_train_small()
validInputs, validTargets = load_valid()
testInputs, testTargets = load_test()

kList = [1, 3, 5, 7, 9]
classRates = range(0, len(kList))
classRatesT = range(0, len(kList))
listCount = 0

for k in kList:
	correctCount = 0
	validLables = run_knn(k, trainInputs, trainTargets, validInputs)
	for i in xrange(len(validLables)):
		if validLables[i] == validTargets[i]:
			correctCount += 1
	classRates[listCount] = (correctCount/float(len(validLables)))
	listCount += 1

listCount = 0
for k in kList:
	correctCount = 0
	validLables = run_knn(k, trainInputs, trainTargets, testInputs)
	for i in xrange(len(validLables)):
		if validLables[i] == testTargets[i]:
			correctCount += 1
	classRatesT[listCount] = (correctCount/float(len(validLables)))
	listCount += 1

print(kList)
print(classRates)
print(classRatesT)

plt.bar(kList, classRates, label = 'Validation Set', color = 'r')
plt.xlabel('k')
plt.ylabel('Classification Rates')
plt.title('Change in Classification Rate with k on Validation Set')
plt.savefig('q3_1_Valid_Rates.png')
plt.show()

plt.bar(kList, classRatesT, label = 'Test Set', color = 'b')
plt.xlabel('k')
plt.ylabel('Classification Rates')
plt.title('Change in Classification Rate with k on Test Set')
plt.savefig('q3_1_Test_Rates.png')
plt.show()