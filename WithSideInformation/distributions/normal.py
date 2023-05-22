"""
Class representing an normal distribution, allowing us to sample from it.
"""
from numpy.random import normal
from numpy.random import multivariate_normal
import numpy, math

# Draw a value for normal with mean=mu and variance=tau
def normal_draw(mu,tau):
    sigma = math.sqrt(tau)
    return normal(loc=mu,scale=sigma,size=None)

def multivariate_draw(mu,cov,type=-1):
    '''
    min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    if min_eig < 0:
        cov -= 10 * min_eig * np.eye(*cov.shape)
   # print(type)
    return multivariate_normal(mu,cov, check_valid='ignore')
    '''
    # min_eig = np.min(np.real(np.linalg.eigvals(cov)))
    # if min_eig < 0:
    #     cov -= 10 * min_eig * np.eye(*cov.shape)
    try:
        d = cov.shape[0]
        epsilon = 0.0001
        newcov = cov + epsilon * np.identity(cov.shape[0])
        L = np.linalg.cholesky(newcov)
        u = np.random.normal(loc=0, scale=1, size=d)
        result= mu + np.dot(L, u)
    except:
        result=multivariate_normal(mu, cov, check_valid='ignore')
    return result

def multivariate_draw_v1(mu,cov,type=-1):

    return multivariate_normal(mu, cov, check_valid='ignore')

def normalpdf(x, mean, var):
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


#print("man",normalpdf(3,3,0.1))


# Do 1000 draws and plot them
import matplotlib.pyplot as plt
import numpy as np
mu = -1.
tau = 1./4.
sigma = 1./2.
'''
s = [normal_draw(mu,tau) for i in range(0,1000)] 
s2 = np.random.normal(mu,sigma, 1000)
count, bins, ignored = plt.hist(s, 50, normed=True)
count, bins, ignored = plt.hist(s2, 50, normed=True)
plt.show()

'''
