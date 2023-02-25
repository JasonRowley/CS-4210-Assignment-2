#-------------------------------------------------------------------------
# AUTHOR: Jason Rowley
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
dbTrain = []
with open('weather_training.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	for i, row in enumerate(reader):
		if i > 0: #skipping the header
			dbTrain.append(row)
            

#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
feature_vals = [
	['Sunny', 'Hot', 'High', 'Weak'],
	['Overcast', 'Mild', 'Normal', 'Strong'],
	['Rain', 'Cool']
]
for row in dbTrain:
	new_row = []
	for j, fv in enumerate(row):
		if j == len(row) - 1:
			# Skip class values column
			continue
		for i, fvs in enumerate(feature_vals):
			if fv in fvs: # conveniently, this will skip the Day column without being explicitly told to because of the defined feature values
				new_row.append(i + 1)
	X.append(new_row)

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
class_vals = ['Yes', 'No']
for row in dbTrain:
	for i, cv in enumerate(class_vals):
		if row[-1] == cv:
			Y.append(i + 1)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
header = []
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
	reader = csv.reader(csvfile)
	for i, row in enumerate(reader):
		if i == 0: 
			header = row
		else:
        		dbTest.append(row)

#printing the header os the solution
#--> add your Python code here
header += ['Confidence']
print(('{:<12} ' * (len(header) - 1) + '{:<12}').format(*(header)))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
probs = []
for row in dbTest:
	new_row = []
	for j, fv in enumerate(row):
		if j == len(row) - 1:
			# Skip class values column
			continue
		for i, fvs in enumerate(feature_vals):
			if fv in fvs:
				new_row.append(i + 1)
	probs.append(list(clf.predict_proba([new_row])[0]))

def argmax(l):
	return l.index(max(l))

confidence = 0.75
to_print = [dbTest[i][:-1] + [class_vals[argmax(p)], max(p)] for i, p in enumerate(probs) if max(p) >= confidence]
for row in to_print:
	print(('{:<12} ' * (len(row) - 1) + '{:<12}').format(*row))
