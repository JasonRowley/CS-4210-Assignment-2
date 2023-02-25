#-------------------------------------------------------------------------
# AUTHOR: Jason Rowley
# FILENAME: decision_tree_2.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

import matplotlib.pyplot as plt

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    feature_vals = [
        ['Young', 'Myope', 'Yes', 'Reduced'],
        ['Prepresbyopic', 'Hypermetrope', 'No', 'Normal'],
        ['Presbyopic']
    ] # the row index is the number for the feature value - 1

    for row in dbTraining:
        new_row = []
        for j, fv in enumerate(row):
            if j == len(row) - 1:
            # Skip class values column
                continue
            for i, fvs in enumerate(feature_vals):
                if fv in fvs:
                    new_row.append(i + 1)
        X.append(new_row)
    # X =

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    class_vals = ['Yes', 'No']
    for row in dbTraining:
        for i, cv in enumerate(class_vals):
                if row[-1] == cv:
                    Y.append(i + 1)
    # Y =

    #loop your training and test tasks 10 times here
    accuracies = []
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       dbTest = []
       X_test = []
       Y_test = []
       num_err = 0
           

       with open('contact_lens_test.csv', 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTest.append (row)
       # dbTest =

       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            numeric_row = []
            
            for j, fv in enumerate(data):
                if j == len(row) - 1:
                # Skip class values column
                    continue
                for i, fvs in enumerate(feature_vals):
                    if fv in fvs:
                        numeric_row.append(i + 1)
            
            data_val = -1
            for i, cv in enumerate(class_vals):
                if data[-1] == cv:
                    data_val = i + 1
            
            class_predicted = clf.predict([numeric_row])[0]

           #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            if class_predicted != data_val:
                num_err += 1

       accuracies.append(1 - num_err / len(dbTest))

    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_acc = sum(accuracies) / 10

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print('final accuracy when training on ' + ds + ': ' + str(avg_acc))




