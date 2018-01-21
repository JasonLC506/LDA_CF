"""
Implement
{author = {Ding, Zhuoye and Qiu, Xipeng and Zhang, Qi and Huang, Xuanjing},
journal = {IJCAI International Joint Conference on Artificial Intelligence},
title = {{Learning topical translation model for microblog hashtag suggestion}},
year = {2013}
} --- [1]
"""
import numpy as np
import cPickle
from scipy.sparse import lil_matrix
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm

from functions import probNormalize, multinomial, logfactorial, probNormalizeLog, logfactorialSparse, expConstantIgnore

np.seterr(divide='raise')

class TTM(object):
    def __init__(self, K):
        """
        :param K: # topics
        """
        # model hyperparameters #
        self.alpha = 0.1                                        # topic distribution prior
        self.beta = 0.01                                        # topic-word distribution prior
        self.gamma = 0.1                                        # topic-emotion distribution prior
        self.delta = 0.01                                       # background-vs-topic distribution prior

        # data dimensions #
        self.E = 0                                              # number of emotions
        self.K = K                                              # number of topics
        self.D = 0                                              # number of documents
        self.Nd = []                                            # number of words of documents (varying over docs)
        self.Md = []                                            # number of emotions of documents (varying over docs)
        self.V = 0                                              # size of vocabulary

        # model latent variables #
        self.theta = None                                       # corpus-level topic distribution [self.K]
        self.pi = None                                          # background-vs-topic distribution
        self.eta = None                                         # topic-emotion distribution [self.K, self.E]
        self.phiB = None                                        # background word distribution [self.V]
        self.phiT = None                                        # topic-word distribution [self.K, self.V]
        self.z = None                                           # document-level topic [self.D]
        self.y = None                                           # word-level background-vs-topic indicator "[self.D, self.Nd]"

        # intermediate variables for fitting #
        self.YI = None                                          # count of background-vs-topic indicator over corpus [2]
        self.TE = None                                          # count of topic-emotion cooccurrences [self.K, self.E]
        self.Y0V = None                                         # count of background word [self.V]
        self.Y1TV = None                                        # count of topic-word cooccurrences [self.K, self.V]
        self.TI = None                                          # count of topic [self.K]
        self.DY1V = None                                        # count of document-level topic-specific word csr_matrix [self.D, self.V]

        # save & restore #
        self.checkpoint_file = "ckpt/TTM"
        self.log_file = "log/TTM"

    def fit(self, dataE, dataW, corpus=None, alpha=0.1, beta=0.01, gamma=0.1, delta=0.01, max_iter=500, resume=None):
        """
        Collapsed Gibbs sampler
        :param dataE: Emotion counts of each document     np.ndarray([self.D, self.E])
        :param dataW: Indexed corpus                      np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """
        self._setHyperparameters(alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        if corpus is None:
            dataToken = self._matrix2corpus(dataW=dataW)
        else:
            dataToken = corpus
        if type(dataW) != np.ndarray:
            pass
            # dataW = dataW.toarray()                             # calculation using np.ndarray rather than scipy.sparse.csr_matrix

        self._setDataDimension(dataE=dataE, dataW=dataW, dataToken=dataToken)
        if resume is None:
            self._initialize(dataE=dataE, dataW=dataW, dataToken=dataToken)
        else:
            self._restoreCheckPoint(filename=resume)
        self._intermediateParameterInitialize(dataE=dataE, dataW=dataW, dataToken=dataToken)

        ppl_initial = self._ppl(dataE=dataE, dataW=dataW, dataToken=dataToken)
        print "before training, ppl: %s" % str(ppl_initial)

        ## Gibbs Sampling ##
        for epoch in range(max_iter):
            self._GibbsSamplingLocal(dataE=dataE, dataW=dataW, dataToken=dataToken, epoch=epoch)
            self._estimateGlobal(dataE)
            ppl = self._ppl(dataE=dataE, dataW=dataW, dataToken=dataToken)
            print "epoch: %d, ppl: %s" % (epoch, str(ppl))
            self._saveCheckPoint(epoch, ppl)

    def _setHyperparameters(self, alpha, beta, gamma, delta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    """ copied from ETM"""
    def _matrix2corpus(self, dataW):
        start = datetime.now()

        dataToken = []
        for d in range(dataW.shape[0]):
            docW = dataW.getrow(d)
            docToken = []
            for w_id in docW.indices:
                w_freq = docW[0, w_id]
                for i in range(w_freq):
                    docToken.append(w_id)
            dataToken.append(docToken)

        duration = datetime.now() - start
        print "_matrix2corpus() takes %fs" % duration.total_seconds()
        return dataToken

    def _setDataDimension(self, dataE, dataW, dataToken):
        self.E = dataE.shape[1]
        self.D = dataE.shape[0]
        self.Nd = map(lambda x: len(x), dataToken)
        self.V = dataW.shape[1]
        self.Md = np.sum(dataE, axis=1).tolist()

    def _initialize(self, dataE, dataW, dataToken):
        start = datetime.now()

        self.theta = probNormalize(np.random.random([self.K]))
        self.pi = probNormalize(np.random.random([2]))
        self.eta = probNormalize(np.random.random([self.K, self.E]))
        self.phiB = probNormalize(np.random.random([self.V]))
        self.phiT = probNormalize(np.random.random([self.K, self.V]))
        self.z = np.zeros([self.D], dtype=np.int8)
        self.y = []
        for d in range(self.D):
            self.z[d] = multinomial(self.theta)
            Nd = self.Nd[d]
            self.y.append(multinomial(self.pi, Nd))

        duration = datetime.now() - start
        print "_initialize() takes %fs" % duration.total_seconds()

    def _intermediateParameterInitialize(self, dataE, dataW, dataToken):

        self.YI = np.zeros([2], dtype=np.int32)
        self.Y0V = np.zeros([self.V], dtype=np.int32)
        self.Y1TV = np.zeros([self.K, self.V], dtype=np.int32)
        self.TE = np.zeros([self.K, self.E], dtype=np.int32)
        self.TI = np.zeros([self.K], dtype=np.int32)
        self.DY1V = lil_matrix((self.D, self.V), dtype=np.int8)

        for d in range(self.D):
            self.TE[self.z[d], :] += dataE[d]
            self.TI[self.z[d]] += 1
            docToken = dataToken[d]
            doc_z = self.z[d]
            doc_y = self.y[d]
            for n in range(self.Nd[d]):
                w = docToken[n]
                w_y = doc_y[n]
                self.YI[w_y] += 1
                if w_y == 0:
                    self.Y0V[w] += 1
                elif w_y == 1:
                    self.Y1TV[doc_z, w] += 1
                    self.DY1V[d, w] += 1
                else:
                    print w_y, type(w_y)
                    raise ValueError("w_y type error")
        # self.DY1V = self.DY1V.tocsr()

    def _GibbsSamplingLocal(self, dataE, dataW, dataToken, epoch):
        """
        Gibbs sampling word-level background-vs-topic and document-level topic
        """
        pbar = tqdm(range(self.D),
                    total = self.D,
                    desc='({0:^3})'.format(epoch))
        for d in pbar:                                 # sequentially sampling
            doc_Nd = self.Nd[d]
            docE = dataE[d]
            docW = dataW[d]
            docToken = dataToken[d]
            doc_z = self.z[d]

            # intermediate parameters calculation #
            Y1T = np.sum(self.Y1TV, axis=1)

            for n in range(doc_Nd):
                w = docToken[n]
                w_y = self.y[d][n]

                ## sampling for y ##
                # calculate leave-one out statistics #
                YI_no_dn_y, Y0V_no_dn_y, Y1TV_no_dn_y = self.YI, self.Y0V, self.Y1TV
                Y1T_no_dn_y = Y1T

                YI_no_dn_y[w_y] += -1
                if w_y == 0:
                    Y0V_no_dn_y[w] += -1
                elif w_y == 1:
                    Y1TV_no_dn_y[doc_z, w] += -1
                    Y1T_no_dn_y[doc_z] += -1
                    self.DY1V[d, w] += -1                           # delete w_y == 1 word
                else:
                    raise ValueError("w_y not 0 or 1")
                # conditional probability #
                prob_w_y_unnorm = np.zeros([2],dtype=np.float32)
                prob_w_y_unnorm[0] = (self.delta + YI_no_dn_y[0]) * (self.beta + Y0V_no_dn_y[w]) / \
                              (self.V * self.beta + YI_no_dn_y[0])
                prob_w_y_unnorm[1] = (self.delta + YI_no_dn_y[1]) * (self.beta + Y1TV_no_dn_y[doc_z, w]) / \
                              (self.V * self.beta + Y1T_no_dn_y[doc_z])
                prob_w_y = probNormalize(prob_w_y_unnorm)
                # new sampled result #
                try:
                    w_y_new = multinomial(prob_w_y)
                except ValueError, e:
                    print prob_w_y_unnorm
                    print prob_w_y
                    print np.sum(prob_w_y), np.sum(prob_w_y) > 1.0
                    print YI_no_dn_y, self.YI, Y0V_no_dn_y[w], Y1TV_no_dn_y[doc_z,w], Y1T_no_dn_y[doc_z]
                    print d
                    raise e
                # update #
                self.y[d][n] = w_y_new
                YI_no_dn_y[w_y_new] += 1
                if w_y_new == 0:
                    Y0V_no_dn_y[w] += 1
                elif w_y_new == 1:
                    Y1TV_no_dn_y[doc_z, w] += 1
                    Y1T_no_dn_y[doc_z] += 1
                    self.DY1V[d, w] += 1                            # add back word with w_y_new == 1
                else:
                    raise ValueError("w_y not 0 or 1")
                self.YI, self.Y0V, self.Y1TV = YI_no_dn_y, Y0V_no_dn_y, Y1TV_no_dn_y
                Y1T = Y1T_no_dn_y

            doc_Y1V = self.DY1V.getrow(d).tocsr()
            doc_Y1V_array = doc_Y1V.toarray().squeeze()
            ## sampling for z ##
            # calculate leave-one out statistics #
            TE_no_d_z, Y1TV_no_d_z, TI_no_d_z = self.TE, self.Y1TV, self.TI

            TE_no_d_z[doc_z,:] += -docE
            Y1TV_no_d_z[doc_z,:] += -doc_Y1V_array
            TI_no_d_z[doc_z] += -1
            # conditional probability #
            prob_doc_z = self._prob_doc_z(TE_no_d_z, Y1TV_no_d_z, TI_no_d_z, docE, docW, doc_Y1V)
            # new sampled result #
            doc_z_new = multinomial(prob_doc_z)
            # update #
            self.z[d] = doc_z_new
            TE_no_d_z[doc_z_new,:] += docE
            Y1TV_no_d_z[doc_z_new, :] += doc_Y1V_array
            TI_no_d_z[doc_z_new] += 1
            self.TE, self.Y1TV, self.TI = TE_no_d_z, Y1TV_no_d_z, TI_no_d_z

    def _estimateGlobal(self, dataE):
        self.theta = probNormalize(self.TI + self.alpha)
        self.pi = probNormalize(self.YI + self.delta)
        self.phiB = probNormalize(self.Y0V + self.beta)
        self.phiT = probNormalize(self.Y1TV + self.beta)
        self.eta = probNormalize(self.TE + self.gamma)

    # def _ppl(self, dataE, dataW, dataToken):
    #     # ppl for word #
    #     log_ppl_w = - (np.inner(self.Y0V, np.log(self.phiB)) +
    #                    np.tensordot(self.Y1TV, np.log(self.phiT), axes=([0,1],[0,1]))) / sum(self.Nd)
    #     # ppl for emotion #
    #     log_ppl_e = - np.tensordot(self.TE, np.log(self.eta), axes=([0,1],[0,1])) / sum(self.Md)
    #     # ppl #
    #     log_ppl = log_ppl_w + log_ppl_e
    #     return log_ppl_w, log_ppl_e, log_ppl_w + log_ppl_e, np.exp(log_ppl)

    def _ppl(self, dataE, dataW, dataToken):
        start = datetime.now()
        self._log("start _ppl")

        ppl_w_log = 0
        ppl_e_log = 0
        ppl_log = 0
        for d in range(self.D):
            docdata = [d, dataToken[d], dataE[d]]
            try:
                doc_ppl_log = self._ppl_log_single_document(docdata)
            except FloatingPointError as e:
                self._log("encounting underflow problem, no need to continue")
                return np.nan, np.nan, np.nan
            ppl_w_log += doc_ppl_log[0]
            ppl_e_log += doc_ppl_log[1]
            ppl_log += doc_ppl_log[2]
        # normalize #
        ppl_w_log /= (sum(self.Nd))
        ppl_e_log /= (sum(self.Md))
        ppl_log /= self.D

        duration = (datetime.now() - start).total_seconds()
        self._log("_ppl takes %fs" % duration)

        return ppl_w_log, ppl_e_log, ppl_log  # word & emoti not separable

    def _ppl_log_single_document(self, docdata):            ### potential underflow problem
        d, docToken, doc_e = docdata
        prob_w_kv = (self.phiT * self.pi[1] + self.phiB * self.pi[0])
        ppl_w_k_log = -np.sum(np.log(prob_w_kv[:, docToken]), axis=1)
        ppl_w_k_scaled, ppl_w_k_constant = expConstantIgnore(- ppl_w_k_log, constant_output=True) # (actual ppl^(-1))

        prob_e_k = self.eta
        ppl_e_k_log = - np.dot(np.log(prob_e_k), doc_e)
        ppl_e_k_scaled, ppl_e_k_constant = expConstantIgnore(- ppl_e_k_log, constant_output=True) # (actual ppl^(-1))
        prob_k = self.theta


        # for emoti given words
        prob_e =  probNormalize(np.tensordot(prob_e_k, np.multiply(prob_k, ppl_w_k_scaled), axes=(0,0)))
        ppl_e_log = - np.dot(np.log(prob_e), doc_e)
        # for words given emoti ! same prob_w for different n
        prob_w = probNormalize(np.tensordot(prob_w_kv, np.multiply(prob_k, ppl_e_k_scaled), axes=(0,0)))
        ppl_w_log = - np.sum(np.log(prob_w[docToken]))
        # for both words & emoti
        try:
            ppl_log = - (np.log(np.inner(ppl_w_k_scaled, np.multiply(ppl_e_k_scaled, prob_k)))
                         + ppl_w_k_constant + ppl_e_k_constant)
        except FloatingPointError as e:
            raise e
        return ppl_w_log, ppl_e_log, ppl_log

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta": self.theta,
            "pi": self.pi,
            "eta": self.eta,
            "phiT": self.phiT,
            "phiB": self.phiB,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "z": self.z,
            "y": self.y,
            "epoch": epoch,
            "ppl": ppl
        }
        with open(filename, "w") as f_ckpt:
            cPickle.dump(state, f_ckpt)

    def _restoreCheckPoint(self, filename=None):
        if filename is None:
            filename = self.checkpoint_file
        state = cPickle.load(open(filename, "r"))
        # restore #
        self.theta = state["theta"]
        self.pi = state["pi"]
        self.eta = state["eta"]
        self.phiT = state["phiT"]
        self.phiB = state["phiB"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.gamma = state["gamma"]
        self.delta = state["delta"]
        self.z = state["z"]
        self.y = state["y"]
        epoch = state["epoch"]
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, epoch, str(ppl))

    def _prob_doc_z(self, TE_no_d_z, Y1TV_no_d_z, TI_no_d_z, docE, docW, doc_Y1V):
        """
        calculate conditional probability for document-level topic doc_z
        method is different from [1]
        :param TE_no_d_z: TE without doc d
        :param Y1TV_no_d_z: Y1TV without d
        :param TI_no_d_z: TI without d
        :param docE: document-level emotion counts [self.E], np.sum(docE) == Md
        :param docW: document-level words counts [self.V], np.sum(docW) == Nd
        :return: doc_z
        """
        # alpha #
        log_prob_a = np.log(TI_no_d_z + self.alpha)
        # TE #
        a = TE_no_d_z + self.gamma
        log_prob_b = np.sum(logfactorial(docE, a), axis=1) - logfactorial(np.sum(docE), np.sum(a, axis=1))
        # Y1TV #
        b = Y1TV_no_d_z + self.beta
        log_prob_c = np.sum(logfactorialSparse(doc_Y1V, b), axis=1) - logfactorial(doc_Y1V.sum(), np.sum(b, axis=1))

        prob_doc_z = probNormalizeLog(log_prob_a + log_prob_b + log_prob_c)
        return prob_doc_z

    def _log(self, string):
        with open(self.log_file, "a") as logf:
            logf.write(string.rstrip("\n") + "\n")

# if __name__ == "__main__":
#     from scipy.sparse import csr_matrix
#     b = np.ones([20,1000],dtype=np.float32)
#     docW = csr_matrix(([2,3,4,5], ([0,0,0,0],[3,4,5,9])),shape=[1,1000])
#     print type(docW.toarray())
#     c = (logfactorial(docW, b))
#     print c.shape
#     print type(c), type(b)
#     print np.sum(c, axis=1).shape
#     # d = np.ones([20,1])
#     # e = np.ones([20])
#     # f = d-e
#     # print f, f.shape