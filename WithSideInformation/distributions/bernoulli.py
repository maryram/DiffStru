"""
Class representing an normal distribution, allowing us to sample from it.
"""
from numpy.random import binomial
import numpy, math


# Draw a value for normal with mean=mu and variance=tau
def bernoulli_draw(p):
    return binomial(1,p)


'''
# Do 60 draws and plot them
import matplotlib.pyplot as plt
import numpy as np
p=2/3
s = [bernoulli_draw(p) for i in range(0,1000)]
count, bins, ignored = plt.hist(s, 60, normed=True)
plt.show()
'''