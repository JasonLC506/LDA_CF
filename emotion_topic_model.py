"""
Implement Bao, S., Xu, S., Zhang, L., Yan, R., Su, Z., Han, D. and Yu, Y., 2012. Mining social emotions from affective text. IEEE transactions on knowledge and data engineering, 24(9), pp.1658-1670.
"""
import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import cPickle

from functions import probNormalize, multinomial

class ETM(object):
    def __init__(self, K):
        """
        :param K: # topics
        """
        # model hyperparameters #
        self.alpha = 0.1                                        # emotion-topic distribution prior
        self.beta = 0.01                                        # topic-word distribution prior

        # data dimensions #
        self.E = 0                                              # number of emotions
        self.K = K                                              # number of topics
        self.D = 0                                              # number of documents
        self.Nd = []                                            # number of words of documents (varying over docs)
        self.V = 0                                              # size of vocabulary

        # model latent variables #
        self.theta = None                                       # emotion-topic distribution [self.E, self.K]
        self.phi = None                                         # topic-word distribution [self.K, self.V]
        self.esp = None                                         # word-level emotion "[self.D, self.Nd]"
        self.z = None                                           # word-level topic   "[self.D, self.Nd]"

        # intermediate variables for fitting #
        self.TE = None                                          # count of topic-emotion cooccurrence [self.K, self.E]
        self.TV = None                                          # count of topic-word cooccurrence [self.K, self.V]
        self.TI = None                                          # count of topic [self.K], np.sum(self.TV, axis=1)
        self.IE = None                                          # count of emotions [self.E], np.sum(self.TE, axis=0)

        # save & restore #
        self.checkpoint_file = "ckpt/ETM"

    def fit(self, dataE, dataW, corpus=None, alpha=0.1, beta=0.01, max_iter = 500, resume = None):
        """
        Collapsed Gibbs sampler
        :param dataE: Emotion distribution of each document     np.ndarray([self.D, self.E])
        :param dataW: Indexed corpus                            np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """
        self._setHyperparameters(alpha=alpha, beta=beta)
        if corpus is None:
            dataToken = self._matrix2corpus(dataW=dataW)
        else:
            dataToken = corpus
        self._setDataDimension(dataE=dataE, dataW=dataW, dataToken=dataToken)
        if resume is None:
            self._initialize(dataE=dataE, dataW=dataW, dataToken=dataToken)
        else:
            self._restoreCheckPoint(filename=resume)

        ppl_initial = self._ppl(dataE=dataE, dataW=dataW, dataToken=dataToken)
        print "before training, ppl: %s" % str(ppl_initial)

        ## Gibbs Sampling ##
        for epoch in range(max_iter):
            self._GibbsSamplingLocal(dataE=dataE, dataW=dataW, dataToken=dataToken, epoch=epoch)
            self._estimateGlobal(dataE)
            ppl = self._ppl(dataE=dataE, dataW=dataW, dataToken=dataToken)
            print "epoch: %d, ppl: %s" % (epoch, str(ppl))
            self._saveCheckPoint(epoch, ppl)

    def _setHyperparameters(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

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

    def _initialize(self, dataE, dataW, dataToken):
        start = datetime.now()

        self.theta = probNormalize(np.random.random([self.E, self.K]))
        self.phi = probNormalize(np.random.random([self.K, self.V]))
        self.esp = []
        self.z = []
        z_dist = np.sum(self.theta, axis=0) / self.E
        for d in range(self.D):
            Nd = self.Nd[d]
            gamma = dataE[d]
            self.esp.append(multinomial(gamma, Nd))
            self.z.append(multinomial(z_dist, Nd))

        self.TE = np.zeros([self.K, self.E], dtype=np.int32)
        self.TV = np.zeros([self.K, self.V], dtype=np.int32)
        for d in range(self.D):
            docToken = dataToken[d]
            doc_z = self.z[d]
            doc_esp = self.esp[d]
            for n in range(self.Nd[d]):
                w = docToken[n]
                w_z = doc_z[n]
                w_esp = doc_esp[n]
                self.TE[w_z, w_esp] += 1
                self.TV[w_z, w] += 1
        self.TI = np.sum(self.TV, axis=1)
        self.IE = np.sum(self.TE, axis=0)

        duration = datetime.now() - start
        print "_initialize() takes %fs" % duration.total_seconds()

    def _GibbsSamplingLocal(self, dataE, dataW, dataToken, epoch):
        """
        Gibbs sampling word-level emotion and topic
        """
        pbar = tqdm(range(self.D),
                    total = self.D,
                    desc='({0:^3})'.format(epoch))
        for d in pbar:                                 # sequentially sampling
            doc_Nd = self.Nd[d]
            docE = dataE[d]
            docToken = dataToken[d]
            for n in range(doc_Nd):
                w = docToken[n]
                w_z = self.z[d][n]
                w_esp = self.esp[d][n]

                ## sampling ##
                # calculate leave-one out statistics #
                TE_no_dn, TV_no_dn, TI_no_dn, IE_no_dn,  = self.TE, self.TV, self.TI, self.IE
                TE_no_dn[w_z, w_esp] += -1
                TV_no_dn[w_z, w] += -1
                TI_no_dn[w_z] += -1
                IE_no_dn[w_esp] += -1
                # conditional probability #
                prob_w_esp = np.divide(np.multiply((self.alpha + TE_no_dn[w_z]), docE),
                                       (self.K * self.alpha + IE_no_dn))
                prob_w_esp = probNormalize(prob_w_esp)
                prob_w_z = np.divide(np.multiply((self.alpha + TE_no_dn[:, w_esp]), (self.beta + TV_no_dn[:, w])),
                                     (self.V * self.beta + TI_no_dn))

                prob_w_z = probNormalize(prob_w_z)
                # new sampled result #
                w_esp_new = multinomial(prob_w_esp)
                w_z_new = multinomial(prob_w_z)
                # update #
                self.z[d][n] = w_z_new
                self.esp[d][n] = w_esp_new
                TE_no_dn[w_z_new, w_esp_new] += 1
                TV_no_dn[w_z_new, w] += 1
                TI_no_dn[w_z_new] += 1
                IE_no_dn[w_esp_new] += 1
                self.TE, self.TV, self.TI, self.IE = TE_no_dn, TV_no_dn, TI_no_dn, IE_no_dn

    def _estimateGlobal(self, dataE):
        self.theta = probNormalize(self.alpha + np.transpose(self.TE))
        self.phi = probNormalize(self.beta + self.TV)

    def _ppl(self, dataE, dataW, dataToken):
        prob_dw = probNormalize(np.tensordot(np.tensordot(dataE, self.theta, axes=(-1,0)), self.phi, axes=(-1,0)))
        ppl = - np.sum(dataW.multiply(np.log(prob_dw)))/sum(self.Nd)
        return ppl, np.exp(ppl)

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta": self.theta,
            "phi": self.phi,
            "alpha": self.alpha,
            "beta": self.beta,
            "esp": self.esp,
            "z": self.z,
            "TE": self.TE,
            "TV": self.TV,
            "TI": self.TI,
            "IE": self.IE,
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
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.esp = state["esp"]
        self.z = state["z"]
        self.TE = state["TE"]
        self.TV = state["TV"]
        self.TI = state["TI"]
        self.IE = state["IE"]
        epoch = state["epoch"]
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, epoch, str(ppl))


if __name__ == "__main__":
    a = np.arange(6).reshape([2,3]).astype(np.float32)
    print np.sum(a, axis=1, keepdims=True)
    print probNormalize(a)