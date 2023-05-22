import numpy as np

def vec(A):
    """
    vector operation
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    return np.reshape(A, (A.size, 1), order='F')

def unvec(A, shape):
    """ 
    brings back a vector to its original shape
    """
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    return np.reshape(A, shape, order='F')

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def get_nonnegative_normal(shape):
    x = np.random.randn(*shape)
    x[x < 0] = 0
    return x


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def isPSD(A, tol=1e-8):
  E = np.linalg.eigvalsh(A)
  return np.all(E > -tol)



#I don't think there is a library which returns the matrix you want, but here is a "just for fun" coding of neareast positive semi-definite matrix algorithm from Higham (2000)


def _getAplus(A):
    eigval, eigvec = np.linalg.eig(A)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    return Q*xdiag*Q.T

def _getPs(A, W=None):
    W05 = np.matrix(W**.5)
    return  W05.I * _getAplus(W05 * A * W05) * W05.I

def _getPu(A, W=None):
    Aret = np.array(A.copy())
    Aret[W > 0] = np.array(W)[W > 0]
    return np.matrix(Aret)

def nearPD(A, nit=10):
    n = A.shape[0]
    W = np.identity(n)
# W is the matrix used for the norm (assumed to be Identity matrix here)
# the algorithm should work for any diagonal W
    deltaS = 0
    Yk = A.copy()
    for k in range(nit):
        Rk = Yk - deltaS
        Xk = _getPs(Rk, W=W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk, W=W)
    return Yk