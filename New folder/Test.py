#! /usr/bin/env python

import numpy as np

a = np.array([1,2,3])
b = np.array([1,3,4])

print(np.square(a+b))

print(np.square(a+b)[1:])

