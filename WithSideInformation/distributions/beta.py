"""
Class representing an normal distribution, allowing us to sample from it.
"""
from numpy.random import beta
import numpy, math


def beta_draw(a,b):
    return beta(a,b)


'''
# Do 60 draws and plot them
import matplotlib.pyplot as plt
import numpy as np
a=3
b=4
s = [beta_draw(a,b) for i in range(0,1000)]
count, bins, ignored = plt.hist(s, 60, normed=True)
plt.show()
'''
