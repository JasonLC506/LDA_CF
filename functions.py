import numpy as np
from scipy.special import gamma, gammaln
from scipy.sparse import csr_matrix
from datetime import datetime

np.seterr(over="raise")

def probNormalize(distributions):
    if distributions.ndim > 1:
        return distributions / np.sum(distributions, axis=1, keepdims=True)
    else:
        return distributions / np.sum(distributions)

def probNormalizeLog(distributions):
    """ from log unnormalized distribution to normalized distribution """
    if distributions.ndim > 1:
        distributions = distributions - np.max(distributions, axis=1,keepdims=True)
    else:
        distributions = distributions - np.max(distributions)
    try:
        prob = probNormalize(np.exp(distributions))
    except FloatingPointError, e:
        print distributions
        print np.max(distributions), np.min(distributions)
        raise e
    return prob

def multinomial(prob, size=1):
    return np.argmax(np.random.multinomial(1, prob, size), axis=1)

def multivariateBeta_inv(x):
    """
    calculate inverse multivariate beta function, as normalization factor for dirichlet distribution
    :param x: np.ndarray(n, d)
    """
    a = np.sum(gammaln(x), axis=1)
    b = gammaln(np.sum(x, axis=1))
    logresult = b - a
    # print "############# multivariateBeta_inv #################"
    # print "eta", x
    # print "a", a
    # print "b", b
    # print "logresult", logresult
    # print "####################################################"
    result = np.exp(logresult)
    return result

def logfactorial(n_factors, start = 1.0):
    """
    stably fast in time in terms of different inputs
    """
    return gammaln(start + n_factors) - gammaln(start)

def logfactorialSparse(n_factors, start = np.array([[1,1],[1,1]])):
    """
    logfactorial when n_factors is sparse and start.shape[0] is small
    :param n_factors: scipy.sparse.csr_matrix shape = [1, V]
    :param start: np.ndarray([K, V]), K is not large
    """
    if start.ndim > 1:
        results = [logfactorialSparse(n_factors, start[k]) for k in range(start.shape[0])]
        return np.array(results)
    else:
        # start: np.ndarray([V]) #
        indices, values = n_factors.indices, n_factors.data
        # print indices
        # print values
        result = np.zeros(start.shape[0], dtype=np.float32)
        for i in range(n_factors.nnz):
            v = indices[i]
            value = values[i]
            result[v] = logfactorial(value, start[v])
        return result

# def logfactorial2(n_factors, start = 1):
#     """
#     ! do not support array input !
#     though a little bit faster with small n_factors,
#     too slow when n_factors is large
#     """
#     factors = np.arange(n_factors, dtype=np.float32)
#     factors += start
#     return np.sum(np.log(factors))

if __name__ == "__main__":
    b = csr_matrix(multinomial([0.999,0.0009,0.0001],20000).reshape([1,20000]))
    a = np.random.random([20, 20000]) * 100
    c = b + a
    print c.shape, type(c)
    # start = datetime.now()
    # for i in range(1000):
    #     r = logfactorialSparse(b, a)
    # # print r
    # print (datetime.now() - start).total_seconds()
