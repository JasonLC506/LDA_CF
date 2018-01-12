"""
Implement
{author = {Zhu, Chen and Zhu, Hengshu and Ge, Yong and Chen, Enhong and Liu, Qi},
journal = {ICDM},
title = {{Tracking the Evolution of Social Emotions: A Time-Aware Topic Modeling Perspective}},
year = {2014}
} --- [1]
"""
import numpy as np
import cPickle
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import dirichlet

from functions import probNormalize, multinomial, multivariateBeta_inv
from emotion_topic_model import ETM

SMOOTH_FACTOR = 0.0001

class eToT(ETM):
    def __init__(self, K):
        """
        :param K: # topics
        """
        super(eToT, self).__init__(K)
        # model hyperparameters #
        self.alpha = 0.1                                        # topic distribution prior
        self.beta = 0.01                                        # topic-word distribution prior

        # data dimensions #
        self.E = 0                                              # number of emotions
        self.K = K                                              # number of topics
        self.D = 0                                              # number of documents
        self.Nd = []                                            # number of words of documents (varying over docs)
        self.V = 0                                              # size of vocabulary

        # model latent variables #
        self.theta = None                                       # document-topic distribution [self.D, self.K]
        self.phi = None                                         # topic-word distribution [self.K, self.V]
        self.eta = None                                         # topic-emotion distribution prior [self.K, self.E]
        self.z = None                                           # word-level topic "[self.D, self.Nd]"

        # intermediate variables for fitting #
        self.TI = None                                          # document-level count of topic [self.D, self.K]
        self.TV = None                                          # count of topic-word cooccurrence [self.K, self.V]
        self.eta_beta_inv = None                                # topic-level inverse multivariate beta function value of self.eta

        # save & restore #
        self.checkpoint_file = "ckpt/eToT"

    # fit(self, dataE, dataW, corpus=None, alpha=0.1, beta=0.01, max_iter = 500)

    # _setHyperparameters(self, alpha, beta)

    # _matrix2corpus(self, dataW)

    # _setDataDimension(self, dataE, dataW, dataToken)

    def _initialize(self, dataE, dataW, dataToken):
        start = datetime.now()

        self.theta = probNormalize(np.random.random([self.D, self.K]))
        self.phi = probNormalize(np.random.random([self.K, self.V]))
        self.eta = np.random.random([self.K, self.E])
        self.z = []
        for d in range(self.D):
            z_dist = self.theta[d]
            Nd = self.Nd[d]
            self.z.append(multinomial(z_dist, Nd))
        self.eta_beta_inv = multivariateBeta_inv(self.eta)

        self.TI = np.zeros([self.D, self.K], dtype=np.int32)
        self.TV = np.zeros([self.K, self.V], dtype=np.int32)
        for d in range(self.D):
            docToken = dataToken[d]
            doc_z = self.z[d]
            for n in range(self.Nd[d]):
                w = docToken[n]
                w_z = doc_z[n]
                self.TI[d, w_z] += 1
                self.TV[w_z, w] += 1

        duration = datetime.now() - start
        print "_initialize() takes %fs" % duration.total_seconds()

    def _GibbsSamplingLocal(self, dataE, dataW, dataToken, epoch):
        """
        Gibbs sampling word-level topic
        """
        pbar = tqdm(range(self.D),
                    total = self.D,
                    desc='({0:^3})'.format(epoch))
        for d in pbar:                                 # sequentially sampling
            doc_Nd = self.Nd[d]
            docE = probNormalize(dataE[d] + SMOOTH_FACTOR)
            docToken = dataToken[d]
            for n in range(doc_Nd):
                w = docToken[n]
                w_z = self.z[d][n]

                ## sampling ##
                # calculate leave-one-out statistics #
                TI_no_dn, TV_no_dn = self.TI, self.TV
                TI_no_dn[d, w_z] += -1
                TV_no_dn[w_z, w] += -1
                # conditional probability #
                prob_pa = TI_no_dn[d] + self.alpha
                prob_pb = np.divide(TV_no_dn[:, w] + self.beta, np.sum(TV_no_dn + self.beta, axis=1))
                prob_pc = np.multiply(self.eta_beta_inv, np.prod(np.power(docE, self.eta - 1), axis=1))
                prob_w_z = probNormalize(prob_pa * prob_pb * prob_pc)
                # new sampled result #
                w_z_new = multinomial(prob_w_z)
                # update #
                self.z[d][n] = w_z_new
                TI_no_dn[d, w_z_new] += 1
                TV_no_dn[w_z_new, w] += 1
                self.TI, self.TV = TI_no_dn, TV_no_dn

    def _estimateGlobal(self, dataE):
        self.theta = probNormalize(self.TI + self.alpha)
        self.phi = probNormalize(self.TV + self.beta)
        self.eta = self._etaUpdate(dataE)
        self.eta_beta_inv = multivariateBeta_inv(self.eta)

    def _etaUpdate(self, dataE):
        """
        use standard MLE estimation of eta from dirichlet distribution
        observation is dataE for each word with word-level topic
        """
        dataE_smoothed = probNormalize(dataE + SMOOTH_FACTOR)
        eta_est = np.zeros([self.K, self.E])
        for k in range(self.K):
            obs = np.repeat(dataE_smoothed, self.TI[:,k].tolist(), axis=0)
            eta_est[k] = dirichlet.mle(obs)
        return eta_est

    # def _etaUpdate(self, dataE):
    #     """
    #     update eta, doubtful
    #     """
    #     T_norm = 1.0 / probNormalize(np.sum(self.TI, axis=0) + SMOOTH_FACTOR)
    #     E_avg = np.dot(np.transpose(self.TI * T_norm), dataE)
    #     E_var = np.zeros([self.K, self.E])
    #     for k in range(self.K):
    #         E_var[k] = T_norm[k] * np.dot(self.TI[:,k], np.power(dataE - E_avg[k,:], 2.0))
    #     eta_est = np.multiply(E_avg, E_avg*(1.0 - E_avg)/E_var - 1.0)                       # different from eq.7 [1]
    #     return eta_est

    def _ppl(self, dataE, dataW, dataToken):
        Nd_total = sum(self.Nd)
        ppl_word = - np.sum(np.multiply(self.TV, np.log(self.phi))) / Nd_total
        ln_dirichlet = np.tensordot(np.log(probNormalize(dataE + SMOOTH_FACTOR)), self.eta - 1, axes=(-1,-1)) + np.log(self.eta_beta_inv)
        ppl_emot = - np.sum(np.multiply(self.TI, ln_dirichlet)) / Nd_total
        return ppl_word, ppl_emot, ppl_word + ppl_emot, np.exp(ppl_emot + ppl_word)

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta": self.theta,
            "phi": self.phi,
            "eta": self.eta,
            "alpha": self.alpha,
            "beta": self.beta,
            "z": self.z,
            "TV": self.TV,
            "TI": self.TI,
            "epoch": epoch,
            "ppl": ppl
        }
        with open(filename, "w") as f_ckpt:
            cPickle.dump(state, f_ckpt)

    def _restoreCheckPoint(self, filename = None):
        if filename is None:
            filename = self.checkpoint_file
        state = cPickle.load(open(filename, "r"))
        # restore #
        self.theta = state["theta"]
        self.phi = state["phi"]
        self.eta = state["eta"]
        self.eta_beta_inv = multivariateBeta_inv(self.eta)
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.z = state["z"]
        self.TV = state["TV"]
        self.TI = state["TI"]
        epoch = state["epoch"]
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, epoch, str(ppl))

def rowMutliply(a, b):
    """
    multiply tensor a with vector b,
    :return: c[i,*] = a[i,*] * b[i]
    """
    return np.transpose(np.transpose(a)*b)