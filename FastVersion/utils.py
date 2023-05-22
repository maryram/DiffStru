import numpy as np

def rmse(x, y, prec=3):
    ans = np.sqrt(np.mean(np.power(x - y, 2)))
    return np.round(ans, prec)

class Normalizer:
    def __init__(self, arr):
        self.weights = np.max(arr, axis=0).T
        
    def fit(self, arr):
        new_arr = np.copy(arr)
        return np.nan_to_num(new_arr / self.weights)
        new_arr[new_arr == np.nan] = 0
        return new_arr
    
    def transform(self, arr):
        return arr * self.weights

def jaccard_cascade(c):
    m = c.shape[0]
    activity = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            intersect = np.sum(((c[j, :] * c[i, :]) > 0) & (c[j, :] > c[i, :]))
            union = np.sum((c[j, :] + c[i, :]) > 0)
            activity[i, j] = intersect / union
    return activity

def is_diagonal(matrix):
    
    # input matrix must be squared
    assert matrix.shape[0] == matrix.shape[1]

    return np.all(matrix * np.eye(matrix.shape[0]) == matrix)

def is_block_diagonal(matrix, d):

    # input matrix must be squared
    assert matrix.shape[0] == matrix.shape[1]

    matrix_p = np.copy(matrix)
    for i in range(matrix.shape[0] // d):
        matrix_p[i*d: i*d+d, i*d:i*d+d] = 0
    return np.sum(matrix_p) == 0

def inverse_block_matrix(matrix, d):
    assert is_block_diagonal(matrix, d)

    matrix_p = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        i1 = i * d
        i2 = i1 + d
        matrix_p[i1:i2, i1:i2] = np.linalg.pinv(matrix[i1:i2, i1:i2])
    return matrix_p

def kron_arr2identity(a, n):
    """
    kronecker product of identity matrix with `n` row in matrix `a`
    """

    """
    Time Complexity: n1 * n2 * n
    n1 * n2 * n^2: for memory allocation (can be reduced to n1 * n2 * n in case of using sparse repr.)
    n1 * n2 * n: there are n1 * n2 block with n elements in diagnal which have to fill
    """

    assert a.ndim == 2

    
    n1, n2 = a.shape
    ans = np.zeros((n1 * n, n2 * n))
    
    for i in range(n1):
        for j in range(n2):
            ans[i * n: i * n + n, j * n: j * n + n] = np.eye(n) * a[i, j]

    return ans
    

def get_sparsity(arr):
    return 1 - np.sum(arr != 0) / arr.size

    

def kron_identity2arr(n, a):
    """
    kronecker product of matrix `a` in identity matrix with `n` rows
    """

    """
    Time Complexity: n1 * n2 * n
    n1 * n2 * n^2: for memory allocation (can be reduced to n1 * n2 * n in case of using sparse repr.)
    n1 * n2 * n: there are n1 * n2 block with n elements in diagnal which have to fill
    """

    assert a.ndim == 2

    n1, n2 = a.shape
    ans = np.zeros((n1 * n, n2 * n))

    for i in range(n):
        ans[i * n1: i * n1 + n1, i * n2: i * n2 + n2] = a.copy()

    return ans

def get_zero_time(matrix):
    min_time = min(matrix[matrix > 0])
    count_zero_times = np.sum(matrix == min_time, axis=0)
    
    #assert count_zero_times.all(), "Some cascades (%d) do not have zero_time" % np.sum(count_zero_times != 0)

    return min_time

def get_second_node_indice(matrix):
    truth_matrix = (np.zeros_like(matrix) == 1)
    for j, cascade in enumerate(matrix.T):
        try:
            time = sorted(set(cascade))[2]
            index = np.where(cascade == time)[0][0]
            truth_matrix[index, j] = True
        except:
            continue
    return truth_matrix
