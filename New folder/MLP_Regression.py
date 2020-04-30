#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
import Network as net
import Data


def Main(argv):

    max_epoc = 100 
    

    data_set = Data.Regression_Dataset("Data/Regression_Data.csv", has_header = False)

    
    data_set.Process_Data()
    
    training_x, training_y = data_set.GetTrainingData()
    test_x, test_y = data_set.GetTestData()
    

    mlp = net.MLP(3, 1, 10, activation = 'linear')

    mlp.InsertHiddenLayer(10)
    mlp.InsertHiddenLayer(20)
    mlp.InsertHiddenLayer(25)
    mlp.InsertHiddenLayer(35)
    mlp.InsertHiddenLayer(40)
    mlp.InsertHiddenLayer(35)
    mlp.InsertHiddenLayer(25)
    mlp.InsertHiddenLayer(15)
    mlp.InsertHiddenLayer(10)
    mlp.InsertHiddenLayer(65)

    error = 1
    j = 0
    while (error > 0.05 and j < max_epoc):
        j += 1
        error = 0
        for i in range(0, training_x.shape[0]):
            output = mlp.ForwardPropagation(training_x[i])
            error += mlp.CalculateError(output, training_y[i])
            mlp.BackwardPropagation(training_x[i], training_y[i])

        error /= training_x.shape[0]
        sys.stderr.write("Training epoch = "+str(j)+" . Error = "+str(error)+"\n")

    #mlp.PrintWeights()

    print("Test result ::\n")
    error = 0
    output = []
    for i in range(0, test_x.shape[0]):
        output_val = mlp.ForwardPropagation(test_x[i])
        output.append(mlp.ForwardPropagation(test_x[i]))
        error += mlp.CalculateError(output_val, test_y[i])
        
    print("Out sample error = "+str(error/test_x.shape[0])+"\n")
    output = np.array(output)
    output = data_set.GetOriginalY(output)

    test_y = data_set.GetOriginalY(test_y)

    for i in range(0, test_x.shape[0]):
        print("Test data ("+str(i)+"). Network output = "+str(output[i])+"\t Actual output = "+str(test_y[i]))


#Script starts from here
if __name__ == "__main__":
    Main(sys.argv)
