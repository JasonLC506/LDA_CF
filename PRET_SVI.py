"""
Personal Reader Emotion Topic model,
extended from TTM
stochastic variational inference
@article{
author = {Hoffman, Matt and Blei, David M. and Wang, Chong and Paisley, John},
title = {{Stochastic Variational Inference}},
journal = {JMLR}
year = {2012}
} --- [2]
jpz5181@ist.psu.edu
"""
import numpy as np
from datetime import datetime
from tqdm import tqdm
import cPickle
import math
from multiprocessing import Pool
import warnings
from copy import copy

from dataDUE_generator import dataDUELoader
from functions import EDirLog, probNormalize

np.seterr(divide='raise', over='raise')

class latentVariableGlobal(object):
    def __init__(self):
        self.data = None                                        # np.ndarray
        self.bigamma_data = None                                # EDirLog(self.data), diff of two digamma functions called bigamma

    def initialize(self, shape=None, new_data=None):
        if new_data is not None:
            self.data = new_data
        else:
            self.data = (np.random.random(shape) + 3.0) / 300.0
        self.bigamma_data = EDirLog(self.data)

    def update(self, new_data, lr):
        self.data = self.data + lr * (new_data - self.data)     # update rule
        self.bigamma_data = EDirLog(self.data)

    def save_state(self):
        return self.data

    def restore_state(self, new_data):
        self.initialize(new_data)



class PRET_SVI(object):
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

        # model latent variables priors #
        # global #
        self.theta = latentVariableGlobal()                     # corpus-level topic distribution [self.K]
        self.pi = latentVariableGlobal()                        # background-vs-topic distribution
        self.eta = latentVariableGlobal()                       # topic-emotion distribution [self.K, self.G, self.E]
        self.phiB = latentVariableGlobal()                      # background word distribution [self.V]
        self.phiT = latentVariableGlobal()                      # topic-word distribution [self.K, self.V]
        self.psi = latentVariableGlobal()                       # user-group distribution [self.U, self.G]
        # local #
        self.z = None                                           # document-level topic [self.D, self.K]                 ## !!! different from SVI original [2], save z for globally to speedup inner iteration
        # self.y = None                                           # word-level background-vs-topic indicator "[self.D, self.Nd, 2]"
        # self.x = None                                           # emoticon-level group indicator "[self.D, self.Md, self.G]"

        # stochastic learning #
        self.lr = None                                          # learning rate pars

        # save & store #
        self.checkpoint_file = "ckpt/PRET_SVI"

        # multiprocess #
        self.pool = None

    def fit(self, dataDUE, dataW, corpus=None, alpha=0.1, beta=0.01, gamma=0.1, delta=0.01, zeta=0.1, max_iter=500, resume=None,
            batch_size=1, N_workers=4, lr_tau=1, lr_kappa=0.9, lr_init=0.1):
        """
        stochastic variational inference
        :param dataDUE: data generator for each document id, generate [[reader_id], [emoticon]]
        :param dataW: Indexed corpus                    np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """
        self._setHyperparameters(alpha, beta, gamma, delta, zeta)
        if corpus is None:
            dataToken = self._matrix2corpus(dataW=dataW)
        else:
            dataToken = corpus

        self._setDataDimension(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)

        self.lr = {"tau": lr_tau, "kappa": lr_kappa, "init": lr_init}

        if resume is None:
            self._initialize(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)
        else:
            self._restoreCheckPoint(filename=resume)
        # self._intermediateParameterInitialize(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)

        # set up multiprocessing pool #
        self.pool = Pool(processes=N_workers)

        ppl_initial = self._ppl(dataDUE, dataW=dataW, dataToken=dataToken)
        print "before training, ppl: %s" % str(ppl_initial)

        for epoch in range(max_iter):
            self._fit_single_epoch(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken, epoch=epoch, batch_size=batch_size)
            ppl = self._ppl(dataDUE, dataW=dataW, dataToken=dataToken)
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
        self.z = probNormalize(np.random.random([self.D, self.K]))
        # self.y = []
        # self.x = []
        # for d in range(self.D):
        #     self.y.append(probNormalize(np.random.random([self.Nd[d], 2])))
        #     self.x.append(probNormalize(np.random.random([self.Md[d], self.G])))

        self.theta.initialize(shape=[self.K])
        self.pi.initialize(shape=[2])
        self.phiB.initialize(shape=[self.V])
        self.phiT.initialize(shape=[self.K, self.V])
        self.psi.initialize(shape=[self.U, self.G])
        self.eta.initialize(shape=[self.K, self.G, self.E])

    def _fit_single_epoch(self, dataDUE, dataW, dataToken, epoch, batch_size):
        # uniformly sampling all documents once #
        pbar = tqdm(dataDUE.batchGenerate(batch_size=batch_size),
                    total = math.ceil(self.D * 1.0 / batch_size),
                    desc = '({0:^3})'.format(epoch))
        for i_batch, batch_size_real, data_batched in pbar:
            local_pars = []
            var_temp = self._fit_batchIntermediateInitialize()
            returned_cursor = self.pool.imap(self._fit_single_document, data_batched)
            for returned in returned_cursor:
                var_temp = self._fit_single_batch_cumulate(returned, var_temp)

            self._fit_single_batch_global_update(var_temp, batch_size_real, epoch)


    def _fit_batchIntermediateInitialize(self):
        vars = {
            "TI": np.zeros(self.K, dtype=np.float64),
            "YI": np.zeros(2, dtype=np.float64),
            "Y0V": np.zeros(self.V, dtype=np.float64),
            "UX": np.zeros([self.U, self.G], dtype=np.float64),
            "Y1TV": np.zeros([self.K, self.V], dtype=np.float64),
            "TXE": np.zeros([self.K, self.G, self.E], dtype=np.float64)
        }
        return vars

    def _fit_single_document(self, docdata, max_iter_inner=500):
        """
        alternative optimization for local parameters for single document
        :return: [d, doc_z, doc_YI, doc_Y0V, doc_UX, doc_Y1zV, doc_zXE]
        """
        d, docToken, [doc_u, doc_e] = docdata
        doc_z = self.z[d]
        doc_Nd = self.Nd[d]
        doc_Md = self.Md[d]
        doc_x_old = np.zeros([doc_Md, self.G])
        doc_y_old = np.zeros([doc_Nd, 2])
        doc_z_old = doc_z
        for inner_iter in xrange(max_iter_inner):
            doc_y = self._fit_single_document_y_update(doc_z, docToken)
            doc_x = self._fit_single_document_x_update(doc_z, doc_u, doc_e)
            doc_z = self._fit_single_document_z_update(doc_y, doc_x, docToken, doc_e)
            doc_x_old, doc_y_old, doc_z_old, converge_flag = self._fit_single_document_convergeCheck(
                                                                    doc_x, doc_y, doc_z, doc_x_old, doc_y_old, doc_z_old
            )
            if converge_flag:
                return self._fit_single_document_return(doc_x, doc_y, doc_z, docToken, doc_u, doc_e)
        warnings.warn("Runtime warning: %d document not converged after %d" % (d, max_iter_inner))
        return self._fit_single_document_return(doc_x, doc_y, doc_z, docToken, doc_u, doc_e)

    def _fit_single_batch_cumulate(self, returned_fit_single_document, var_temp):
        d, doc_z, doc_YI, doc_Y0V, doc_UX, doc_Y1zV, doc_zXE = returned_fit_single_document  # parse returned from self._fit_single_document

        # update document-level topic #
        self.z[d] = doc_z

        var_temp["TI"][doc_z] += 1
        var_temp["YI"] += doc_YI
        var_temp["Y0V"] += doc_Y0V
        var_temp["UX"] += doc_UX
        var_temp["Y1TV"][doc_z, :] += doc_Y1zV
        var_temp["TXE"][doc_z, :, :] += doc_zXE

    def _fit_single_batch_global_update(self, var_temp, batch_size_real, epoch):
        lr = self._lrCal(epoch)
        batch_weight = self.D * 1.0 / batch_size_real
        new_theta_temp = self.alpha + batch_weight * var_temp["TI"]
        new_pi_temp = self.delta + batch_weight * var_temp["YI"]
        new_phiB_temp = self.beta + batch_weight * var_temp["Y0V"]
        new_phiT_temp = self.beta + batch_weight * var_temp["Y1TV"]
        new_psi_temp = self.zeta + batch_weight * var_temp["UX"]
        new_eta_temp = self.gamma + batch_weight * var_temp["TXE"]
        self.theta.update(new_theta_temp, lr)
        self.pi.update(new_pi_temp, lr)
        self.phiB.update(new_phiB_temp, lr)
        self.phiT.update(new_phiT_temp, lr)
        self.psi.update(new_psi_temp, lr)
        self.eta.update(new_eta_temp, lr)

    def _lrCal(self, epoch):
        return float(self.lr["init"] * np.power((self.lr["tau"] + epoch), - self.lr["kappa"]))

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        start = datetime.now()

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
            # "y": self.y,
            # "x": self.x,
            "epoch": epoch,
            "ppl": ppl
        }
        with open(filename, "w") as f_ckpt:
            cPickle.dump(state, f_ckpt)

        duration = datetime.now() - start
        print "_saveCheckPoint takes %f s" % duration.total_seconds()

    def _restoreCheckPoint(self, filename=None):
        start = datetime.now()

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
        # self.y = state["y"]
        # self.x = state["x"]
        epoch = state["epoch"]
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, epoch, str(ppl))

        duration = datetime.now() - start
        print "_restoreCheckPoint takes %f s" % duration.total_seconds()


