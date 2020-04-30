#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
import Network as net
import Data
import matplotlib.pyplot as plt

def Test_Train(data_set, train_data_frac):
    
    max_epoc = 1000 
    
    data_set.Reset_process()

    data_set.Process_Data(frac = train_data_frac) 
    
    training_x, training_y = data_set.GetTrainingData()
    test_x, test_y = data_set.GetTestData()

    mlp = net.MLP(training_x.shape[1], training_y.shape[1], 5)

    mlp.InsertHiddenLayer(training_x.shape[1]+10)
    mlp.InsertHiddenLayer(training_x.shape[1]+15)
    mlp.InsertHiddenLayer(int((training_x.shape[1]+training_y.shape[1])/2) + 10)
    mlp.InsertHiddenLayer(training_y.shape[1]+15)
    mlp.InsertHiddenLayer(training_y.shape[1]+10)
    
    error = 1
    j = 0
    while (error > 0.05 and j < max_epoc):
        error = 0
        j += 1
        for i in range(0, training_x.shape[0]):
            output = mlp.ForwardPropagation(training_x[i])
            error += mlp.CalculateError(output, training_y[i])
            mlp.BackwardPropagation(training_x[i], training_y[i])

        error /= training_x.shape[0]

    error = 0
    for i in range(0, test_x.shape[0]):
        output = mlp.ForwardPropagation(test_x[i])
        error += mlp.CalculateError(output, test_y[i])
    
    return error/test_x.shape[0]

def Main(argv):

    data_set = Data.Classification_DataSet("Data/nursery.data", has_header = False)
    
    train_data_frac = []
    error = []
    frac = 0.3
    for i in range (0,10):
        train_data_frac.append(frac)
        error.append(Test_Train(data_set, frac))
        frac += 0.05
        sys.stderr.write("For training data fraction = "+str(frac)+"\t error = "+str(error[len(error)-1])+"\n")
    
    plt.plot(train_data_frac, error)
    plt.xlabel('Traning data fraction')
    plt.ylabel('Out sample error')
    plt.show()


#Script starts from here
if __name__ == "__main__":
    Main(sys.argv)
