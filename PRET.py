"""
Personal Reader Emotion Topic model,
extended from TTM
jpz5181@ist.psu.edu
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.special import gammaln
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import cPickle

from functions import probNormalize, multinomial, logfactorial, probNormalizeLog, logfactorialSparse
from dataDUE_generator import dataDUE

np.seterr(divide='raise')

class PRET(object):
    def __init__(self, K, G):
        """
        :param K: # topics
        :param G: # groups
        """
        # model hyperparameters #
        self.alpha = 0.1                                        # topic distribution prior
        self.beta = 0.01                                        # topic-word distribution prior
        self.gamma = 0.1                                        # (topic * group)-emotion distribution prior
        self.delta = 0.01                                       # background-vs-topic distribution prior
        self.zeta = 0.1                                         # user-group distribution

        # data dimension #
        self.E = 0                                              # number of emotions
        self.K = K                                              # number of topics
        self.G = G                                              # number of groups
        self.D = 0                                              # number of documents
        self.Nd = []                                            # number of words of documents (varying over docs)
        self.Md = []                                            # number of emotions of documents (varying over docs)
        self.V = 0                                              # size of vocabulary
        self.U = 0                                              # number of users

        # model latent variables #
        self.theta = None                                       # corpus-level topic distribution [self.K]
        self.pi = None                                          # background-vs-topic distribution
        self.eta = None                                         # topic-emotion distribution [self.K, self.G, self.E]
        self.phiB = None                                        # background word distribution [self.V]
        self.phiT = None                                        # topic-word distribution [self.K, self.V]
        self.psi = None                                         # user-group distribution [self.U, self.G]
        self.z = None                                           # document-level topic [self.D]
        self.y = None                                           # word-level background-vs-topic indicator "[self.D, self.Nd]"
        self.x = None                                           # emoticon-level group indicator "[self.D, self.Md]"

        # intermediate variables for fitting #
        self.YI = None                                          # count of background-vs-topic indicator over corpus [2]
        self.Y0V = None                                         # count of background word [self.V]
        self.Y1TV = None                                        # count of topic-word cooccurrences [self.K, self.V]
        self.TI = None                                          # count of topic [self.K]
        self.TXE = None                                         # count of topic-group-emotion cooccurrences [self.K, self.G, self.E]
        self.UX = None                                          # count of user-group cooccurrences [self.U, self.G]
        self.DY1V = None                                        # count of document-level topic-specific word csr_matrix [self.D, self.V]
        self.DXE = None                                         # count of document-level group-emotion cooccurrences [self.G, self.E]

        # save & restore #
        self.checkpoint_file = "ckpt/PRET"

    def fit(self, dataDUE, dataW, corpus=None, alpha=0.1, beta=0.01, gamma=0.1, delta=0.01, zeta=0.1, max_iter=500, resume=None):
        """
        Collapsed Gibbs sampler
        :param dataDUE: data generator for each document id, generate [[reader_id], [emoticon]]
        :param dataW: Indexed corpus                    np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """
        self._setHyperparameters(alpha, beta, gamma, delta, zeta)
        if corpus is None:
            dataToken = self._matrix2corpus(dataW=dataW)
        else:
            dataToken = corpus

        self._setDataDimension(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)
        if resume is None:
            self._initialize(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)
        else:
            self._restoreCheckPoint(filename=resume)
        self._intermediateParameterInitialize(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)

        ppl_initial = self._ppl(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)
        print "before training, ppl: %s" % str(ppl_initial)

        ## Gibbs Sampling ##
        for epoch in range(max_iter):
            self._GibbsSamplingLocal(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken, epoch=epoch)
            self._estimateGlobal(dataDUE)
            ppl = self._ppl(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)
            print "epoch: %d, ppl: %s" % (epoch, str(ppl))
            self._saveCheckPoint(epoch, ppl)

    def _setHyperparameters(self, alpha, beta, gamma, delta, zeta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta

    """ copied from ETM """
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

    def _setDataDimension(self, dataDUE, dataW, dataToken):
        self.E = dataDUE.E
        self.U = dataDUE.U
        self.Md = dataDUE.Md
        self.D = dataW.shape[0]
        self.Nd = map(lambda x: len(x), dataToken)
        self.V = dataW.shape[1]

    def _initialize(self, dataDUE, dataW, dataToken):
        start = datetime.now()

        self.theta = probNormalize(np.random.random([self.K]))
        self.pi = probNormalize(np.random.random([2]))
        self.eta = probNormalize(np.random.random([self.K, self.G, self.E]))
        self.phiB = probNormalize(np.random.random([self.V]))
        self.phiT = probNormalize(np.random.random([self.K, self.V]))
        self.psi = probNormalize(np.random.random([self.U, self.G]))

        self.z = np.zeros([self.D], dtype=np.int8)
        self.y = []
        self.x = []
        for d in range(self.D):
            self.z[d] = multinomial(self.theta)
            self.y.append(multinomial(self.pi, self.Nd[d]))
            doc_x = []
            for m in range(self.Md[d]):
                u = np.random.randint(0,self.U)
                doc_x.append(multinomial(self.psi[u]))
            self.x.append(np.array(doc_x, dtype=np.int8))

        duration = datetime.now() - start
        print "_initialize() takes %fs" % duration.total_seconds()

    def _intermediateParameterInitialize(self, dataDUE, dataW, dataToken):
        self.YI = np.zeros([2], dtype=np.int32)
        self.Y0V = np.zeros([self.V], dtype=np.int32)
        self.Y1TV = np.zeros([self.K, self.V], dtype=np.int32)
        self.TI = np.zeros([self.K], dtype=np.int32)
        self.TXE = np.zeros([self.K, self.G, self.E], dtype=np.int32)
        self.UX = np.zeros([self.U, self.G], dtype=np.int16)
        self.DY1V = lil_matrix((self.D, self.V), dtype = np.int8)
        self.DXE = np.zeros([self.D, self.G, self.E], dtype = np.int32)

        for d, [doc_u, doc_e] in dataDUE.generate(batch_size=1, random_shuffle=False):
            self.TI[self.z[d]] += 1
            docToken = dataToken[d]
            doc_z = self.z[d]
            doc_y = self.y[d]
            doc_x = self.x[d]
            for n in range(self.Nd[d]):
                w = docToken[n]
                w_y = doc_y[n]
                self.YI[w_y] += 1
                if w_y == 0:
                    self.Y0V[w] += 1
                else:
                    self.Y1TV[doc_z, w] += 1
                    self.DY1V[d, w] += 1
            for m in range(self.Md[d]):
                u = doc_u[m]
                e = doc_e[m]
                x = doc_x[m]
                self.TXE[doc_z, x, e] += 1
                self.UX[u, x] += 1
                self.DXE[d, x, e] += 1
        # self.DY1V = self.DY1V.tocsr()

    def _GibbsSamplingLocal(self, dataDUE, dataW, dataToken, epoch):
        """
        Gibbs sampling  word-level background-vs-topic
                        document-level topic
                        emoticon-level group
        """
        pbar = tqdm(dataDUE.generate(batch_size=1, random_shuffle=False),
                    total = self.D,
                    desc = '({0:^3})'.format(epoch))
        for d, [doc_u, doc_e] in pbar:
            docToken = dataToken[d]
            docW = dataW[d]
            doc_Nd = self.Nd[d]
            doc_Md = self.Md[d]

            # update document-level topic #
            self._doc_z_update(d, doc_u, doc_e, docW, docToken)

            # update word-level background-vs-topic #
            Y1T = np.sum(self.Y1TV, axis=1)
            for n in xrange(doc_Nd):
                Y1T = self._y_update(d, n, doc_u, doc_e, docW, docToken, Y1T)

            # update emoticon-level group #
            TX = np.sum(self.TXE, axis=-1)
            for m in xrange(doc_Md):
                TX = self._x_update(d, m, doc_u, doc_e, docW, docToken, TX)

    def _doc_z_update(self, d, doc_u, doc_e, docW, docToken):
        """ update document-level topic """
        doc_z = self.z[d]
        doc_XE = self.DXE[d]
        doc_Y1V = self.DY1V.getrow(d).tocsr()
        doc_Y1V_array = doc_Y1V.toarray().squeeze()

        # calculate leave-one out statistics #
        TI_no_d, TXE_no_d, Y1TV_no_d = self.TI, self.TXE, self.Y1TV
        TI_no_d[doc_z] += -1
        TXE_no_d[doc_z,:,:] += - doc_XE
        Y1TV_no_d[doc_z,:] += - doc_Y1V_array

        # conditional probability #
        prob_doc_z = self._prob_doc_z(TI_no_d, TXE_no_d, Y1TV_no_d, doc_XE, doc_Y1V)

        # new sampled result #
        doc_z_new = multinomial(prob_doc_z)

        # update #
        self.z[d] = doc_z_new
        TI_no_d[doc_z_new] += 1
        TXE_no_d[doc_z_new, :, :] += doc_XE
        Y1TV_no_d[doc_z_new, :, :] += doc_Y1V_array
        self.TI, self.TXE, self.Y1TV = TI_no_d, TXE_no_d, Y1TV_no_d

    def _prob_doc_z(self, TI_no_d, TXE_no_d, Y1TV_no_d, doc_XE, doc_Y1V):
        """
        calculate conditional probability for document-level topic doc_z
        :param: doc_Y1V: lil_matrix((1, self.V), dtype=int8)
        """
        # alpha #
        log_prob_alpha = np.log(TI_no_d + self.alpha)
        # gamma # (without sparsity, directly calculating log gamma function)
        a = TXE_no_d + self.gamma + doc_XE
        log_prob_gamma = np.sum(np.sum(gammaln(a), axis=-1) - gammaln(np.sum(a, axis=-1)), axis=-1)
        # beta # (with sparsity)
        b = Y1TV_no_d + self.beta
        log_prob_beta = np.sum(logfactorialSparse(doc_Y1V, b), axis=-1) - logfactorial(doc_Y1V.sum(), np.sum(b, axis=-1))

        prob_doc_z = probNormalizeLog(log_prob_alpha + log_prob_gamma + log_prob_beta)
        return prob_doc_z

    def _y_update(self, d, n, doc_u, doc_e, docW, docToken, Y1T):
        """
        update word-level background-vs-topic indicator
        """
        w = docToken[n]
        w_y = self.y[d][n]
        doc_z = self.z[d]

        # calculate leave-one out statistics #
        YI_no_dn_y, Y0V_no_dn_y, Y1TV_no_dn_y = self.YI, self.Y0V, self.Y1TV
        Y1T_no_dn_y = Y1T

        YI_no_dn_y[w_y] += -1
        if w_y == 0:
            Y0V_no_dn_y[w] += -1
        else:
            Y1TV_no_dn_y[doc_z, w] += -1
            Y1T_no_dn_y[doc_z] += -1
            self.DY1V[d, w] += -1  # delete w_y == 1 word

        # conditional probability #
        prob_w_y_unnorm = np.zeros([2], dtype=np.float32)
        prob_w_y_unnorm[0] = (self.delta + YI_no_dn_y[0]) * (self.beta + Y0V_no_dn_y[w]) / \
                             (self.V * self.beta + YI_no_dn_y[0])
        prob_w_y_unnorm[1] = (self.delta + YI_no_dn_y[1]) * (self.beta + Y1TV_no_dn_y[doc_z, w]) / \
                             (self.V * self.beta + Y1T_no_dn_y[doc_z])
        prob_w_y = probNormalize(prob_w_y_unnorm)

        # new sampled result #
        w_y_new = multinomial(prob_w_y)

        # update #
        self.y[d][n] = w_y_new
        YI_no_dn_y[w_y_new] += 1
        if w_y_new == 0:
            Y0V_no_dn_y[w] += 1
        else:
            Y1TV_no_dn_y[doc_z, w] += 1
            Y1T_no_dn_y[doc_z] += 1
            self.DY1V[d, w] += 1  # add back word with w_y_new == 1
        self.YI, self.Y0V, self.Y1TV = YI_no_dn_y, Y0V_no_dn_y, Y1TV_no_dn_y
        Y1T = Y1T_no_dn_y
        return Y1T

    def _x_update(self, d, m, doc_u, doc_e, docW, docToken, TX):
        """ update emoticon-level group indicator"""
        doc_z = self.z[d]
        u = doc_u[m]
        e = doc_e[m]
        x = self.x[d][m]

        # calculate leave-one out statistics #
        TXE_no_dm, UX_no_dm, TX_no_dm = self.TXE, self.UX, TX
        TXE_no_dm[doc_z, x, e] += -1
        UX_no_dm[u, x] += -1
        TX_no_dm[doc_z, x] += -1

        # calculate conditional probability #
        prob_gamma = (self.gamma + TXE_no_dm[doc_z,:,e]) / (self.E * self.gamma + TX_no_dm[doc_z, :])
        prob_zeta = self.zeta + UX_no_dm[u, :]
        prob_x = probNormalize(prob_gamma * prob_zeta)

        # new sampled result #
        x_new = multinomial(prob_x)

        # update #
        self.x[d][m] = x_new
        TXE_no_dm[doc_z, x_new, e] += 1
        UX_no_dm[u, x_new] += 1
        TX_no_dm[doc_z, x_new] += 1
        self.TXE, self.UX, TX = TXE_no_dm, UX_no_dm, TX_no_dm
        return TX

    def _estimateGlobal(self, dataDUE=None):
        self.theta = probNormalize(self.alpha + self.TI)
        self.pi = probNormalize(self.delta + self.YI)
        self.phiB = probNormalize(self.Y0V + self.beta)
        self.phiT = probNormalize(self.Y1TV + self.beta)
        self.eta = probNormalize(self.TXE + self.gamma)
        self.psi = probNormalize(self.UX + self.zeta)

    def _ppl(self, dataDUE, dataW, dataToken):
        # ppl for word #
        log_ppl_w = - (np.inner(self.Y0V, np.log(self.phiB)) +
                       np.tensordot(self.Y1TV, np.log(self.phiT), axes=([0, 1], [0, 1]))) / sum(self.Nd)
        # ppl for emoticons #
        log_ppl_e = - np.tensordot(self.TXE, np.log(self.eta), axes = ([0, 1, 2], [0, 1, 2])) / sum(self.Md)
        # ppl #
        log_ppl = log_ppl_w + log_ppl_e
        return log_ppl_w, log_ppl_e, log_ppl, np.exp(log_ppl)

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta": self.theta,
            "pi": self.pi,
            "eta": self.eta,
            "phiT": self.phiT,
            "phiB": self.phiB,
            "psi": self.psi,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "zeta": self.zeta,
            "z": self.z,
            "y": self.y,
            "x": self.x,
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
        self.psi  = state["psi"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.gamma = state["gamma"]
        self.delta = state["delta"]
        self.zeta = state["zeta"]
        self.z = state["z"]
        self.y = state["y"]
        self.x = state["x"]
        epoch = state["epoch"]
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, epoch, str(ppl))