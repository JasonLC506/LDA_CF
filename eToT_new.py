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

from functions import probNormalize, multinomial, multivariateBeta_inv, expConstantIgnore, probNormalizeLog
from dataDUE_generator import dataDUELoader

SMOOTH_FACTOR = 0.0001

class eToT(object):
    def __init__(self, K):
        """
        :param K: # topics
        """
        # model hyperparameters #
        self.alpha = 0.1                                        # topic distribution prior
        self.beta = 0.01                                        # topic-word distribution prior

        # data dimensions #
        self.E = 0                                              # number of emotions
        self.K = K                                              # number of topics
        self.D = 0                                              # number of documents
        self.D_train = 0                                        # number of documents in training
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

        self.dataE_smoothed = {}                                # save dataE_smoothed to accelerate

        # save & restore #
        self.checkpoint_file = "ckpt/period_foxnews/eToT"
        self.log_file = "log/period_foxnews/eToT"

    def fit(self, dataDUE, dataDUE_valid_on_shell=None, dataDUE_valid_off_shell=None, alpha=0.1, beta=0.01, max_iter = 500, resume = None):
        """
        Collapsed Gibbs sampler
        :param dataE: Emotion distribution of each document     np.ndarray([self.D, self.E])
        :param dataW: Indexed corpus                            np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """
        self._setHyperparameters(alpha=alpha, beta=beta)

        self._setDataDimension(dataDUE = dataDUE)
        if resume is None:
            self._initialize(dataDUE=dataDUE)
        else:
            self._restoreCheckPoint(filename=resume)

        if dataDUE_valid_on_shell is None:
            ppl_on_shell = [None, None, None]
        else:
            ppl_on_shell = self._ppl(dataDUE_valid_on_shell, epoch=-1, on_shell=True)

        if dataDUE_valid_off_shell is None:
            ppl_off_shell = [None, None, None]
        else:
            ppl_off_shell = self._ppl(dataDUE_valid_off_shell, epoch=-1, on_shell=False)

        ppl_initial = ppl_on_shell + ppl_off_shell
        self._log("before training, ppl: %s" % str(ppl_initial))

        ## Gibbs Sampling ##
        for epoch in range(max_iter):
            self._GibbsSamplingLocal(dataDUE, epoch=epoch)
            self._estimateGlobal()
            if dataDUE_valid_on_shell is None:
                ppl_on_shell = [None, None, None]
            else:
                ppl_on_shell = self._ppl(dataDUE_valid_on_shell, epoch=-1, on_shell=True)

            if dataDUE_valid_off_shell is None:
                ppl_off_shell = [None, None, None]
            else:
                ppl_off_shell = self._ppl(dataDUE_valid_off_shell, epoch=-1, on_shell=False)

            ppl = ppl_on_shell + ppl_off_shell
            self._log("epoch: %d, ppl: %s" % (epoch, str(ppl)))
            self._saveCheckPoint(epoch, ppl)

    def _setHyperparameters(self, alpha, beta):
        self.alpha = 1.0 / self.K
        self.beta = beta

    def _setDataDimension(self, dataDUE):
        self.E = dataDUE.E
        self.D = dataDUE.D
        self.D_train = dataDUE.D_current_data
        self.Nd = dataDUE.Nd
        self.V = dataDUE.V

    def _initialize(self, dataDUE):
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
        self.dataE_smoothed = {}
        for docdata in dataDUE.generate():
            d, docToken, [doc_u, doc_e] = docdata
            doc_z = self.z[d]
            for n in range(self.Nd[d]):
                w = docToken[n]
                w_z = doc_z[n]
                self.TI[d, w_z] += 1
                self.TV[w_z, w] += 1
            doc_E = np.sum(np.identity(self.E, dtype=np.float64)[:, doc_e], axis=1)
            docE = probNormalize(doc_E + SMOOTH_FACTOR)
            self.dataE_smoothed[d] = docE

        duration = datetime.now() - start
        self._log("_initialize() takes %fs" % duration.total_seconds())

    def _GibbsSamplingLocal(self, dataDUE, epoch):
        """
        Gibbs sampling word-level topic
        """
        pbar = tqdm(dataDUE.generate(),
                    total = self.D_train,
                    desc='({0:^3})'.format(epoch))
        for docdata in pbar:                                 # sequentially sampling
            d, docToken, [doc_u, doc_e] = docdata
            doc_Nd = self.Nd[d]
            if d in self.dataE_smoothed:
                docE = self.dataE_smoothed[d]
            else:
                doc_E = np.sum(np.identity(self.E, dtype=np.float64)[:, doc_e], axis=1)
                docE = probNormalize(doc_E + SMOOTH_FACTOR)
                self.dataE_smoothed[d] = docE
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

    def _estimateGlobal(self):
        self.theta = probNormalize(self.TI + self.alpha)
        self.phi = probNormalize(self.TV + self.beta)
        self.eta = self._etaUpdate()
        self.eta_beta_inv = multivariateBeta_inv(self.eta)

    def _etaUpdate(self):
        """
        use standard MLE estimation of eta from dirichlet distribution
        observation is dataE for each word with word-level topic
        """
        dataE_smoothed = np.zeros([self.D, self.E])
        for d in self.dataE_smoothed:
            dataE_smoothed[d] = self.dataE_smoothed[d]
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

    def _ppl(self, dataDUE, epoch=-1, on_shell=False):
        start = datetime.now()

        ppl_w_log = 0
        ppl_e_log = 0
        ppl_log = 0

        Nd_sum = 0
        Md_sum = 0
        D_sum = 0

        for docdata in dataDUE.generate():
            try:
                if on_shell:
                    doc_ppl_log, Nd, Md = self._ppl_log_single_document_on_shell(docdata)
                else:
                    doc_ppl_log, Nd, Md = self._ppl_log_single_document_off_shell(docdata)
            except FloatingPointError as e:
                raise e
                return np.nan, np.nan, np.nan
            ppl_w_log += doc_ppl_log[0]
            ppl_e_log += doc_ppl_log[1]
            ppl_log += doc_ppl_log[2]
            Nd_sum += Nd
            Md_sum += Md
            D_sum += 1
        # normalize #
        ppl_w_log /= Nd_sum
        ppl_e_log /= Md_sum
        ppl_log /= D_sum

        duration = (datetime.now() - start).total_seconds()
        print "_ppl takes %fs" % duration

        return ppl_w_log, ppl_e_log, ppl_log

    def _ppl_log_single_document_off_shell(self, docdata):
        d, docToken, [doc_u, doc_e] = docdata
        Nd = docToken.shape[0]
        prob_z_alpha = np.ones([self.K, Nd], dtype=np.float64)
        ppl_w_z = self.phi[:, docToken]
        prob_z = prob_z_alpha * ppl_w_z
        prob_z_sum = np.sum(prob_z, axis=1) # product over dirichlet is the same of sum over dirichlet priors
        prob_e = probNormalize(np.dot(prob_z_sum, self.eta))
        ppl_e_log = - np.sum(np.log(prob_e)[doc_e])

        doc_E = np.sum(np.identity(self.E, dtype=np.float64)[:, doc_e], axis=1)
        docE = probNormalize(doc_E + SMOOTH_FACTOR)
        ppl_e_z_log = - (np.tensordot(np.log(docE), self.eta-1.0, axes=(0,1)) + np.log(self.eta_beta_inv))
        ppl_e_z_scaled, ppl_e_z_constant = expConstantIgnore(- ppl_e_z_log, constant_output=True)

        prob_w = probNormalize(np.dot(ppl_e_z_scaled, self.phi))
        ppl_w_log = - np.sum(np.log(prob_w)[docToken])

        ppl_log = np.nan
        return [ppl_w_log, ppl_e_log, ppl_log], docToken.shape[0], doc_e.shape[0]


    def _ppl_log_single_document_on_shell(self, docdata):
        d, docToken, [doc_u, doc_e] = docdata
        doc_z = np.array(self.z[d], dtype=np.int8)
        doc_eta = np.sum(self.eta[doc_z, :], axis=0)
        prob_e = probNormalize(doc_eta)
        ppl_e_log = - np.sum(np.log(prob_e)[doc_e])

        ppl_w_log = - np.sum(np.log(self.phi)[doc_z, docToken])

        ppl_log = ppl_e_log + ppl_w_log

        return [ppl_w_log, ppl_e_log, ppl_log], docToken.shape[0], doc_e.shape[0]

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

    def _log(self, string):
        with open(self.log_file, "a") as logf:
            logf.write(string.rstrip("\n") + "\n")

def rowMutliply(a, b):
    """
    multiply tensor a with vector b,
    :return: c[i,*] = a[i,*] * b[i]
    """
    return np.transpose(np.transpose(a)*b)