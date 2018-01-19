import numpy as np
from scipy.special import gamma, gammaln, digamma
from scipy.sparse import csr_matrix
from datetime import datetime
import warnings

np.seterr(over="raise")

def probNormalize(distributions):
    if distributions.ndim > 1:
        return distributions / np.sum(distributions, axis=-1, keepdims=True)
    else:
        return distributions / np.sum(distributions)

def probNormalizeLog(distributions):
    """ from log unnormalized distribution to normalized distribution """
    try:
        prob = probNormalize(expConstantIgnore(distributions))
    except FloatingPointError, e:
        print distributions
        print np.max(distributions), np.min(distributions)
        raise e
    return prob

def expConstantIgnore(log_array, constant_output = False):
    if log_array.ndim > 1:
        constant = np.max(log_array, axis=-1,keepdims=True)
        log_array = log_array - constant
    else:
        constant = np.max(log_array)
        log_array = log_array - constant
    try:
        array = np.exp(log_array)
    except FloatingPointError as e:
        print log_array
        print np.max(log_array), np.min(log_array)
        raise e
    if constant_output:
        return array, constant
    return array

def multinomial(prob, size=1):
    return np.argmax(np.random.multinomial(1, prob, size), axis=1)

def multinomialSingleUnnorm(probability):
    rnd = np.random.random() * np.sum(probability)
    cpd = 0.0
    for i in xrange(probability.shape[0]):
        cpd += probability[i]
        if rnd <= cpd:
            return i
    # raise RuntimeError("multinomialSingleUnnorm failed with input %s, rnd %f" % (str(probability), rnd))
    warnings.warn("multinomialSingleUnnorm failed with input %s, rnd %f" % (str(probability), float(rnd)))
    return probability.shape[0] - 1

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


def EDirLog(par):
    # E_{Dir(theta; par)}(log(theta))
    return digamma(par) - digamma(np.sum(par, axis=-1, keepdims=True))


if __name__ == "__main__":
    par = np.ones([3,4,5])
    print EDirLog(par)
