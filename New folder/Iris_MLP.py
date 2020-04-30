#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
import Network as net
import Data


def Main(argv):
    
    max_epoc = 1000

    data_set = Data.Classification_DataSet("Data/iris.csv")
    data_set.Process_Data()

    mlp = net.MLP(4, 3, 4)

    mlp.InsertHiddenLayer(20)
    mlp.InsertHiddenLayer(35)
    mlp.InsertHiddenLayer(25)
    mlp.InsertHiddenLayer(15)

    training_x, training_y = data_set.GetTrainingData()
    test_x, test_y = data_set.GetTestData()

    # print(training_x.shape)
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
    for i in range(0, test_x.shape[0]):
        output = mlp.ForwardPropagation(test_x[i])
        error += mlp.CalculateError(output, test_y[i])
        print("Network prediction = "+str(output))
        print("Actual Output = "+str(test_y[i]))
        print("\n")
    
    print("Out sample error = "+str(error/test_x.shape[0]))

#Script starts from here
if __name__ == "__main__":
    Main(sys.argv)
