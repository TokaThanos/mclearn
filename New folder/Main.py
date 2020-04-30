#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
import Network as net
import Data


def Main(argv):

    max_epoc = 100 
    
    Data.Generate_Regression_Data("Data/Regression_Data.csv")


#Script starts from here
if __name__ == "__main__":
    Main(sys.argv)
