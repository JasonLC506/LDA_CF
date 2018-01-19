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
from multiprocessing import Pool, Process
import warnings
from copy import copy
import itertools

from dataDUE_generator import dataDUELoader
from functions import EDirLog, probNormalize, probNormalizeLog, expConstantIgnore
from PRET_SVI_functions import _fit_single_document, _ppl_new_process

np.seterr(divide='raise', over='raise')

tqdm.monitor_interval = 0                                       # workaround tqdm RuntimeError: Set changed size during iteration

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

        # model global latent variables point estimate #
        self.GLV = {"theta": None, "pi": None, "eta": None, "phiB": None, "phiT": None, "psi": None}

        # stochastic learning #
        self.lr = None                                          # learning rate pars

        # save & store #
        self.checkpoint_file = "ckpt/PRET_SVI"
        self.log_file = "log/PRET_SVI"

        # multiprocess #
        self.pool = None
        self.process = None

    def fit(self, dataDUE, dataW, dataDUE_valid=None, corpus=None, alpha=0.1, beta=0.01, gamma=0.1, delta=0.01, zeta=0.1, max_iter=500, resume=None,
            batch_size=1024, N_workers=4, lr_tau=1, lr_kappa=0.1, lr_init=1.0, converge_threshold_inner=0.01):
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
        self.converge_threshold_inner = converge_threshold_inner            # inner iteration for each document

        if resume is None:
            self._initialize(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)
        else:
            self._restoreCheckPoint(filename=resume)
        # self._intermediateParameterInitialize(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken)

        # set up multiprocessing pool #
        # self.pool = Pool(processes=N_workers)

        self._estimateGlobal()
        if dataDUE_valid is None:
            ppl_initial = self._ppl(dataDUE, dataW=dataW, dataToken=dataToken, epoch=-1)
            self._log("before training, ppl: %s" % str(ppl_initial))
        else:
            self._ppl_multiprocess(dataDUE_valid, epoch=-1)

        for epoch in range(max_iter):
            self._fit_single_epoch(dataDUE=dataDUE, dataW=dataW, dataToken=dataToken, epoch=epoch, batch_size=batch_size)
            self._estimateGlobal()
            if dataDUE_valid is None:
                ppl = self._ppl(dataDUE, dataW=dataW, dataToken=dataToken, epoch=epoch)
                self._log("epoch: %d, ppl: %s" % (epoch, str(ppl)))
                self._saveCheckPoint(epoch, ppl)
            else:
                self._ppl_multiprocess(dataDUE_valid, epoch=epoch)

    def _ppl_multiprocess(self, dataDUE_valid, epoch):
        if self.process is not None:
            self.process.join()             # wait until last epoch ppl result completed
        pars_topass = self._fit_single_epoch_pars_topass()
        self.process = Process(target=_ppl_new_process, args=(dataDUE_valid.data_queue, dataDUE_valid.D, pars_topass, epoch,))
        self.process.daemon = True
        self.process.start()

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
        self._log("_matrix2corpus() takes %fs" % duration.total_seconds())
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
        print "start _initialize"

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

        duration = (datetime.now() - start).total_seconds()
        print "_initialize takes %fs" % duration

    def _estimateGlobal(self):
        """
        give point estimate of global latent variables, self.GLV
        current: mean
        """
        self.GLV["theta"] = probNormalize(self.theta.data)
        self.GLV["pi"] = probNormalize(self.pi.data)
        self.GLV["psi"] = probNormalize(self.psi.data)
        self.GLV["phiB"] = probNormalize(self.phiB.data)
        self.GLV["phiT"] = probNormalize(self.phiT.data)
        self.GLV["eta"] = probNormalize(self.eta.data)

    def _fit_single_epoch(self, dataDUE, dataW, dataToken, epoch, batch_size):
        """ single process"""
        self._log("start _fit_single_epoch")
        start = datetime.now()

        # uniformly sampling all documents once #
        pbar = tqdm(dataDUE.batchGenerate(batch_size=batch_size),
                    total = math.ceil(self.D * 1.0 / batch_size),
                    desc = '({0:^3})'.format(epoch))
        for i_batch, batch_size_real, data_batched in pbar:
            var_temp = self._fit_batchIntermediateInitialize()

            pars_topass = self._fit_single_epoch_pars_topass()

            ### test ###
            # returned_cursor = self.pool.imap_unordered(_fit_single_document, data_batched_topass)
            # for returned in returned_cursor:
            for data_batched_sample in data_batched:
                returned = _fit_single_document(data_batched_sample, pars_topass)
                var_temp = self._fit_single_batch_cumulate(returned, var_temp)

            # end3 = datetime.now()###
            self._fit_single_batch_global_update(var_temp, batch_size_real, epoch)
            # end4 = datetime.now()###
            # print "_fit_single_batch_global_update takes %fs" % (end4 - end3).total_seconds()###
        duration = (datetime.now() - start).total_seconds()
        self._log("_fit_single_epoch takes %fs" % duration)

    def _fit_single_epoch_pars_topass(self):
        ans = vars(self)
        pars_topass = {}
        for name in ans:
            if name == "pool":
                continue
            if name == "process":
                continue
            pars_topass[name] = ans[name]
        return pars_topass

    def _ppl(self, dataDUE, dataW, dataToken, epoch=-1):
        start = datetime.now()
        self._log("start _ppl")

        ppl_w_log = 0
        ppl_e_log = 0
        ppl_log = 0
        pbar = tqdm(dataDUE.generate(),
                    total=self.D,
                    desc = '{}'.format("_ppl"))
        for docdata in pbar:
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

        return ppl_w_log, ppl_e_log, ppl_log                                # word & emoti not separable

    def _ppl_log_single_document(self, docdata):            ### potential underflow problem
        d, docToken, [doc_u, doc_e] = docdata
        prob_w_kv = (self.GLV["phiT"] * self.GLV["pi"][1] + self.GLV["phiB"] * self.GLV["pi"][0])
        ppl_w_k_log = -np.sum(np.log(prob_w_kv[:, docToken]), axis=1)
        ppl_w_k_scaled, ppl_w_k_constant = expConstantIgnore(- ppl_w_k_log, constant_output=True) # (actual ppl^(-1))

        prob_e_mk = np.dot(self.GLV["psi"][doc_u, :], self.GLV["eta"])
        ppl_e_k_log = - np.sum(np.log(prob_e_mk[np.arange(doc_u.shape[0]), :, doc_e]), axis=0)
        ppl_e_k_scaled, ppl_e_k_constant = expConstantIgnore(- ppl_e_k_log, constant_output=True) # (actual ppl^(-1))
        prob_k = self.GLV["theta"]


        # for emoti given words
        prob_e_m =  probNormalize(np.tensordot(prob_e_mk, np.multiply(prob_k, ppl_w_k_scaled), axes=(1,0)))
        ppl_e_log = - np.sum(np.log(prob_e_m[np.arange(doc_u.shape[0]), doc_e]))
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

    def _fit_batchIntermediateInitialize(self):
        # instantiate vars every loop #
        vars = {
            "TI": np.zeros(self.K, dtype=np.float64),
            "YI": np.zeros(2, dtype=np.float64),
            "Y0V": np.zeros(self.V, dtype=np.float64),
            "UX": np.zeros([self.U, self.G], dtype=np.float64),
            "Y1TV": np.zeros([self.K, self.V], dtype=np.float64),
            "TXE": np.zeros([self.K, self.G, self.E], dtype=np.float64)
        }
        return vars

    def _fit_single_batch_cumulate(self, returned_fit_single_document, var_temp):
        # ends = []
        # ends.append(datetime.now())###
        # print "#### start _fit_single_batch_cumulate ####"

        d, doc_z, doc_YI, doc_Y0V, doc_u, doc_x, doc_Y1TV, doc_TXE = returned_fit_single_document  # parse returned from self._fit_single_document

        # ends.append(datetime.now())###

        # update document-level topic #
        self.z[d, :] = doc_z[:]

        var_temp["TI"] += doc_z

        # ends.append(datetime.now())###

        var_temp["YI"] += doc_YI

        # ends.append(datetime.now())###

        var_temp["Y0V"] += doc_Y0V

        # ends.append(datetime.now()) ###

        # var_temp["UX"] += doc_UX    # too sparse
        var_temp["UX"][doc_u, :] += doc_x

        # ends.append(datetime.now()) ###

        var_temp["Y1TV"] += doc_Y1TV

        # ends.append(datetime.now()) ###

        var_temp["TXE"] += doc_TXE

        # print "#### _fit_single_batch_cumulate detail profile: read, z, YI, Y0V, UX, Y1TV, TXE: ", [(ends[i]-ends[i-1]).total_seconds() for i in range(1, len(ends))]###

        return var_temp

    def _fit_single_batch_global_update(self, var_temp, batch_size_real, epoch):
        # ends = []###
        # ends.append(datetime.now())    ###

        lr = self._lrCal(epoch)

        # ends.append(datetime.now())###

        batch_weight = self.D * 1.0 / batch_size_real
        new_theta_temp = self.alpha + batch_weight * var_temp["TI"]
        new_pi_temp = self.delta + batch_weight * var_temp["YI"]
        new_phiB_temp = self.beta + batch_weight * var_temp["Y0V"]
        new_phiT_temp = self.beta + batch_weight * var_temp["Y1TV"]
        new_psi_temp = self.zeta + batch_weight * var_temp["UX"]
        new_eta_temp = self.gamma + batch_weight * var_temp["TXE"]

        # ends.append(datetime.now())###

        self.theta.update(new_theta_temp, lr)

        # ends.append(datetime.now())###

        self.pi.update(new_pi_temp, lr)

        # ends.append(datetime.now())###

        self.phiB.update(new_phiB_temp, lr)

        # ends.append(datetime.now())###

        self.phiT.update(new_phiT_temp, lr)

        # ends.append(datetime.now())###

        self.psi.update(new_psi_temp, lr)

        # ends.append(datetime.now())###

        self.eta.update(new_eta_temp, lr)

        # ends.append(datetime.now())###

        # print "_fit_single_batch_global_update, detail profile for ## lr, add, theta, pi, phiB, phiT, psi, eta", [(ends[i] - ends[i-1]).total_seconds() for i in range(1, len(ends))]

    def _lrCal(self, epoch):
        return float(self.lr["init"] * np.power((self.lr["tau"] + epoch), - self.lr["kappa"]))

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        start = datetime.now()

        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta": self.theta.data,
            "pi": self.pi.data,
            "eta": self.eta.data,
            "phiT": self.phiT.data,
            "phiB": self.phiB.data,
            "psi": self.psi.data,
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
        self._log("_saveCheckPoint takes %f s" % duration.total_seconds())

    def _restoreCheckPoint(self, filename=None):
        start = datetime.now()

        if filename is None:
            filename = self.checkpoint_file
        state = cPickle.load(open(filename, "r"))
        # restore #
        self.theta.initialize(new_data=state["theta"])
        self.pi.initialize(new_data=state["pi"])
        self.eta.initialize(new_data=state["eta"])
        self.phiT.initialize(new_data=state["phiT"])
        self.phiB.initialize(new_data=state["phiB"])
        self.psi.initialize(new_data=state["psi"])
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

        # for model display #
        self._estimateGlobal()

    def _log(self, string):
        with open(self.log_file, "a") as logf:
            logf.write(string.rstrip("\n") + "\n")

if __name__ == "__main__":
    model = PRET_SVI(2,1)
    ans = vars(model)
    print type(ans)
    print ans

