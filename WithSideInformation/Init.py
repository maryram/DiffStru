import numpy as np

def common_neighbor(G):
    new_G = np.copy(np.array(G, dtype=np.int64))
    new_G[new_G.T == 1] = 1
    N = new_G.shape[0]

    ans = np.zeros_like(new_G)
    for i in range(N):
        for j in range(N):
            ans[i, j] = sum(new_G[i, :] & new_G[j, :])
    ans = ans - ans * np.eye(N)
    return ans


def common_neighbor_directed(G):
    N = G.shape[0]
    new_G = np.copy(np.array(G, dtype=np.int64))
    for i in range(N):
        for j in range(N):
            if(i!=j):
                new_G[i,j]=1
            else:
                new_G[i,j]=np.sum(np.multiply(G[i,:],G[:,j].T))
    return new_G

def common_cascade(cascade_indicator):
    """
    returns a M*M matrix where i,j entry is the count of common nodes in i-th and j-th cascades
    cascade_indicator: N*M matrix of cascades
    """

    M = cascade_indicator.shape[1]
    ans = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            ans[i, j] = sum(cascade_indicator[:, i] & cascade_indicator[:, j])

    ans = ans - ans * np.eye(M)
    return ans
