# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 22:59:12 2025

@author: ravir
"""

import random
route = list(range(4))
random.shuffle(route)
print(route)


import numpy as np
mat = np.random.rand(9,2) * 10   #this is for getting values till 10 , 2D matrix
print(mat)


tour = [2]
i = tour[0]
print(i)


val = random.random()

print(val)


x = [random.randint(0, 5)for _ in range(5)]
print(x)
