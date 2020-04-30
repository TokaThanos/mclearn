import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import random

class Regression_Dataset:
    
    def __init__(self, csv_file, has_header = True):
        if (has_header):
            self._df = pd.read_csv(csv_file)
        else:
            self._df = pd.read_csv(csv_file, header = None)
        self._test_data = []
        self._training_data = []
        self._scalerX = None
        self._scalerY = None

    def Process_Data(self, frac = 0.66):
        self._df = self._df.sample(frac=1)      # Randomly shuffles the data

        train_data_size = int(self._df.shape[0] * frac)

        x_vector_dimension = self._df.shape[1] - 1      # last column is the class label

        attribute_df = self._df.iloc[:, 0 : x_vector_dimension]
        class_df = self._df.iloc[:, x_vector_dimension : ]

        training_attr_array = attribute_df.iloc[0 : train_data_size, :].values
        training_class_array = class_df.iloc[0 : train_data_size, :].values

        self._scalerX = StandardScaler().fit(training_attr_array)
        self._scalerY = StandardScaler().fit(training_class_array)

        self._training_data.append(self._scalerX.transform(training_attr_array))
        self._training_data.append(self._scalerY.transform(training_class_array))

        test_attr_array = attribute_df.iloc[train_data_size : , :].values
        test_class_array = class_df.iloc[train_data_size : , :].values


        self._test_data.append(self._scalerX.transform(test_attr_array))
        self._test_data.append(self._scalerY.transform(test_class_array))


    def GetTrainingData(self):
        return self._training_data[0], self._training_data[1]
    
    def GetTestData(self):
        return self._test_data[0], self._test_data[1]

    def GetOriginalX(self, X):
        return self._scalerX.inverse_transform(X)

    def GetOriginalY(self, Y):
        return self._scalerY.inverse_transform(Y)




class Classification_DataSet:

    def __init__(self, csv_file, has_header = True):
        if (has_header):
            self._df = pd.read_csv(csv_file)
        else:
            self._df = pd.read_csv(csv_file, header = None)
        self._test_data = []
        self._training_data = []

    def Reset_process(self):
        self._test_data = []
        self._training_data = []

    def Process_Data(self, frac = 0.66):
        self._df = self._df.sample(frac=1)      # Randomly shuffles the data

        train_data_size = int(self._df.shape[0] * frac)

        x_vector_dimension = self._df.shape[1] - 1      # last column is the class label

        attribute_df = self._df.iloc[:, 0 : x_vector_dimension]
        class_df = self._df.iloc[:, x_vector_dimension : ]

        attribute_df = pd.get_dummies(attribute_df)
        class_df = pd.get_dummies(class_df)

        self._training_data.append(attribute_df.iloc[0 : train_data_size, : ].values)
        self._training_data.append(class_df.iloc[0 : train_data_size, :].values)

        self._test_data.append(attribute_df.iloc[train_data_size : , :].values)
        self._test_data.append(class_df.iloc[train_data_size : , :].values)
        
    def GetTrainingData(self):
        return self._training_data[0], self._training_data[1]
    
    def GetTestData(self):
        return self._test_data[0], self._test_data[1]


def NonLinear_Function(x,y,z):
    return math.sin(x) + (math.pi)*math.cos(y) + ((math.pi)**2)*math.tan(z) 

def Generate_Regression_Data(file_name, size = 300):
    f = open(file_name, "w")

    for i in range(size):
        x = random.random()*20 - 10         # range [-10, 10]
        y = random.random()*100 - 50        # range [-50, 50]
        z = random.random()*2 - 1           # range [-1, 1]

        val = NonLinear_Function(x, y, z)
        f.write(str(x)+","+str(y)+","+str(z)+","+str(val)+"\n")

