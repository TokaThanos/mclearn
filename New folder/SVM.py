#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def Main(argv):
    bankdata = pd.read_csv("Data/bill_authentication.csv")

    X = bankdata.drop('Class', axis=1)
    y = bankdata['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)
    
    print("Printing Confusion Matrix::")
    print(confusion_matrix(y_test,y_pred))
    print("\n\n Printing  Classification report::\n")
    print(classification_report(y_test,y_pred))




#Script starts from here
if __name__ == "__main__":
    Main(sys.argv)
