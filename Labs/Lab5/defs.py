import numpy as np
def markov(rien,rng):
    A = np.array([[0.5, 0.5, 0], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    rho = np.array([0.25, 0.75, 0])
    nmax = 100
    n = A.shape[0]
    m = A.shape[1]
    assert n == m , "A is not square"
    assert n == len(rho), "rho and A have different dimensions"
    assert rho.sum() == 1, "rho does not sum to 1"
    assert np.all(A >= 0), "A has negative elements"
    assert np.all(A.sum(axis=1) == 1), "A does not sum to 1"
    np.random.seed(rng)
    X=np.zeros((nmax),dtype=int)
    for k in range(nmax):
        if k == 0:
            X[k] = np.random.choice([i for i in range(1,n+1)],1,p=rho)
        else:
            X[k] = np.random.choice([i for i in range(1,n+1)],1,p=A[X[k-1]-1,:])
            
    return X