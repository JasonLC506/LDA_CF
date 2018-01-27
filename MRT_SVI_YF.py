"""
Personal Multi-Relational Topic Model,
extended from PRET,
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
from MRT_SVI_YF_functions import _fit_single_document

np.seterr(divide='raise', over='raise')

tqdm.monitor_interval = 0                                       # workaround tqdm RuntimeError: Set changed size during iteration

class latentVariableGlobal(object):
    def __init__(self):
        self.data = None                                        # np.ndarray
        self.bigamma_data = None                                # EDirLog(self.data), diff of two digamma functions called bigamma

    def initialize(self, shape=None, seed=None, new_data=None, additional_noise_axis=None):
        """
        :param shape:
        :param seed:
        :param new_data:
        :param additional_noise_axis:
        :return:
        """
        if new_data is not None:
            self.data = new_data
        else:
            noise = (np.random.random(shape) + 3.0) / 300.0
            if additional_noise_axis is not None:
                noise_addition = np.random.random(shape[additional_noise_axis:])/10.0           # special for eta
                noise += noise_addition
            self.data = noise*seed + seed
        self.bigamma_data = EDirLog(self.data)

    def update(self, new_data, lr):
        self.data = self.data + lr * (new_data - self.data)     # update rule
        self.bigamma_data = EDirLog(self.data)

    def save_state(self):
        return self.data

    def restore_state(self, new_data):
        self.initialize(new_data)


class MRT_SVI_YF(object):
    def __init__(self, K, G, A):
        """
        :param K: # topics
        :param G: # groups
        """
        # model hyperparameters #
        self.alpha = 1.0 / K                                    # topic distribution prior [2]
        self.beta = 0.01                                        # topic-word distribution prior
        self.gamma = 1000                                       # (topic * group)-emotion distribution prior
        self.delta = 0.1                                        # background-vs-topic distribution prior
        self.zeta = 0.01                                        # user-group distribution
        self.h = 0.1                                            # background vs topical prior

        # data dimension #
        self.E = 0                                              # number of emotions
        self.K = K                                              # number of topics
        self.G = G                                              # number of groups
        self.A = A                                              # number of attitudes
        self.D = 0                                              # number of documents                                                              ! including off_shell text
        self.D_train = 0                                        # number of documents in training set
        self.Nd = []                                            # number of words of documents (varying over docs)                                 ! including off_shell text
        self.Md = []                                            # number of emotions of documents (varying over docs)                              ! only include training data
        self.V = 0                                              # size of vocabulary
        self.U = 0                                              # number of users

        # model latent variables priors #
        # global #
        self.theta = latentVariableGlobal()                     # corpus-level topic distribution [self.K]
        self.pi = latentVariableGlobal()                        # topic-attitude distribution [self.K, self.A]
        self.piB = latentVariableGlobal()                       # background attitude distribution [self.A]
        self.eta = latentVariableGlobal()                       # attitude-emotion distribution [self.A, self.G, self.E]
        self.phi = latentVariableGlobal()                       # topic-word distribution [self.K, self.V]
        self.psi = latentVariableGlobal()                       # user-group distribution [self.U, self.G]
        self.c = latentVariableGlobal()                         # background vs topical attitude [2]
        # local #
        self.z = None                                           # document-level topic [self.D, self.K]                                             ! including off_shell text, need abandon in prediction
        self.f = None                                           # document-level attitude [self.D, self.A]                                          ! including off_shell text, need abandon in prediction
        self.y = None                                           # document-level background vs topical [self.D, 2]

        # model global latent variables point estimate #
        self.GLV = {"theta": None, "pi": None, "piB": None, "eta": None, "phi": None, "psi": None, "c": None}

        # stochastic learning #
        self.lr = None                                          # learning rate pars

        # inner iteration converge threshold #
        self.converge_threshold_inner = 0.0

        # save & store #
        self.checkpoint_file = "ckpt/MRT_SVI_YF"
        self.epoch_init = 0
        self.lr_step = 0
        self.log_file = "log/MRT_SVI_YF"

        # multiprocess #
        # self.pool = None
        # self.process = None

    def fit(self, dataDUE, dataDUE_valid_on_shell=None, dataDUE_valid_off_shell=None,
            alpha=0.1, beta=0.01, gamma=1000, delta=0.001, zeta=0.01, h=0.1,
            max_iter=500, resume=None,
            batch_size=1024, N_workers=4, lr_tau=2, lr_kappa=0.1, lr_init=1.0, converge_threshold_inner=0.01):
        """
        stochastic variational inference
        :param dataDUE: data generator for each document id, generate [[reader_id], [emoticon]]
        :param dataW: Indexed corpus                    np.ndarray([self.D, self.V]) scipy.sparse.csr_matrix
        """

        self._setDataDimension(dataDUE=dataDUE)

        self._setHyperparameters(alpha, beta, gamma, delta, zeta, h, lr_tau, lr_kappa, lr_init, converge_threshold_inner)

        if resume is None:
            self._initialize(dataDUE=dataDUE)
        else:
            self._restoreCheckPoint(filename=resume)

        # set up multiprocessing pool #
        # self.pool = Pool(processes=N_workers)

        self._estimateGlobal()
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
        ppl_best = ppl_initial

        if batch_size > self.D_train/2:
            batch_size = self.D_train
            self._log("set batch_size as D_train, full batch")

        for epoch in range(self.epoch_init, max_iter):
            self._fit_single_epoch(dataDUE=dataDUE, epoch=epoch, batch_size=batch_size)
            self._estimateGlobal()
            if dataDUE_valid_on_shell is None:
                ppl_on_shell = [None, None, None]
            else:
                ppl_on_shell = self._ppl(dataDUE_valid_on_shell, epoch=epoch, on_shell=True)
            if dataDUE_valid_off_shell is None:
                ppl_off_shell = [None, None, None]
            else:
                ppl_off_shell = self._ppl(dataDUE_valid_off_shell, epoch=epoch, on_shell=False)
            ppl = ppl_on_shell + ppl_off_shell

            ### test ###
            ppl_off_shell_for_on_shell = self._ppl(dataDUE_valid_on_shell, epoch, on_shell=False)
            print "epoch: %d, ppl: %s" % (epoch, str(ppl))
            print "ppl_off_shell for on_shell", str(ppl_off_shell_for_on_shell)

            ppl_best, best_flag = self._ppl_compare(ppl_best, ppl)
            self._log("epoch: %d, ppl: %s" % (epoch, str(ppl)))
            self._saveCheckPoint(epoch, ppl)
            for i in range(len(best_flag)):
                if best_flag[i]:
                    self._saveCheckPoint(epoch=epoch, ppl=ppl, filename=self.checkpoint_file + "_best_ppl[%d]" % i)

    def _setDataDimension(self, dataDUE):
        self.E = dataDUE.E
        self.U = dataDUE.U
        self.Md = dataDUE.Md
        self.D = dataDUE.D
        self.D_train = dataDUE.D_current_data
        self.Nd = dataDUE.Nd
        self.V = dataDUE.V
        print "set data dimension,", "D", self.D, "D_train", self.D_train

    def _setHyperparameters(self, alpha, beta, gamma, delta, zeta, h, lr_tau, lr_kappa, lr_init, converge_threshold_inner):
        self.alpha = 1.0 / self.K        # fixed based on [2]
        self.beta = beta
        # self.gamma = beta * self.V * sum(self.Md) / (1.0 * self.E * sum(self.Nd) * self.G)
        self.gamma = gamma
        self.delta = delta
        self.zeta = zeta
        self.h = h

        self.lr = {"tau": lr_tau, "kappa": lr_kappa, "init": lr_init}

        self.converge_threshold_inner = converge_threshold_inner  # inner iteration for each document

        print "set up hyperparameters: alpha=%f, beta=%f, gamma=%f, delta=%f, zeta=%f, h=%f" % (self.alpha, self.beta, self.gamma, self.delta, self.zeta, self.h)

    def _initialize(self, dataDUE):
        start = datetime.now()
        print "start _initialize"

        self.z = probNormalize(np.random.random([self.D, self.K]))
        self.f = probNormalize(np.random.random([self.D, self.A]))
        self.y = probNormalize(np.random.random([self.D, 2]))
        self.theta.initialize(shape=[self.K], seed=self.alpha)
        self.pi.initialize(shape=[self.K, self.A], seed=self.delta)
        self.piB.initialize(shape=[self.A], seed=self.delta)
        self.phi.initialize(shape=[self.K, self.V], seed=self.beta)
        self.psi.initialize(shape=[self.U, self.G], seed=self.zeta)
        self.eta.initialize(shape=[self.A, self.G, self.E], seed=self.gamma)
        self.c.initialize(shape=[2], seed=self.h)

        duration = (datetime.now() - start).total_seconds()
        print "_initialize takes %fs" % duration

    def _estimateGlobal(self):
        """
        give point estimate of global latent variables, self.GLV
        current: mean
        """
        self.GLV["theta"] = probNormalize(self.theta.data)
        self.GLV["pi"] = probNormalize(self.pi.data)
        self.GLV["piB"] = probNormalize(self.piB.data)
        self.GLV["psi"] = probNormalize(self.psi.data)
        self.GLV["phi"] = probNormalize(self.phi.data)
        self.GLV["eta"] = probNormalize(self.eta.data)
        self.GLV["c"] = probNormalize(self.c.data)

    def _ppl(self, dataDUE, epoch=-1, on_shell=False):
        start = datetime.now()
        self._log("start _ppl")

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
                self._log("encounting underflow problem, no need to continue")
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
        self._log("_ppl takes %fs" % duration)

        return ppl_w_log, ppl_e_log, ppl_log                                # word & emoti not separable

    def _ppl_log_single_document_off_shell(self, docdata):            ### potential underflow problem
        d, docToken, [doc_u, doc_e] = docdata
        prob_w_kv = self.GLV["phi"]
        ppl_w_k_log = -np.sum(np.log(prob_w_kv[:, docToken]), axis=1)
        ppl_w_k_scaled, ppl_w_k_constant = expConstantIgnore(- ppl_w_k_log, constant_output=True) # (actual ppl^(-1))

        prob_e_mk = np.dot(self.GLV["psi"][doc_u, :],
                           (np.tensordot(self.GLV["pi"], self.GLV["eta"], axes=(1,0)) * self.GLV["c"][1] +
                            np.tensordot(self.GLV["piB"], self.GLV["eta"], axes=(0,0)) * self.GLV["c"][0])
                           )
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
        ppl_log = - (np.log(np.inner(ppl_w_k_scaled, np.multiply(ppl_e_k_scaled, prob_k)))
                     + ppl_w_k_constant + ppl_e_k_constant)

        return [ppl_w_log, ppl_e_log, ppl_log], docToken.shape[0], doc_u.shape[0]

    def _ppl_log_single_document_on_shell(self, docdata):
        d, docToken, [doc_u, doc_e] = docdata

        prob_w_kv = self.GLV["phi"]
        prob_w = probNormalize(np.tensordot(prob_w_kv, self.z[d], axes=(0,0)))
        ppl_w_log = -np.sum(np.log(prob_w[docToken]))

        prob_e_mf = np.dot(self.GLV["psi"][doc_u, :], self.GLV["eta"])
        prob_e_m = probNormalize(np.tensordot(prob_e_mf, self.f[d], axes=(1,0)))
        ppl_e_log = -np.sum(np.log(prob_e_m[np.arange(doc_u.shape[0]), doc_e]))

        ppl_log = ppl_w_log + ppl_e_log
        return [ppl_w_log, ppl_e_log, ppl_log], docToken.shape[0], doc_u.shape[0]

    def _fit_single_epoch(self, dataDUE, epoch, batch_size):
        """ single process"""
        self._log("start _fit_single_epoch")
        start = datetime.now()

        # uniformly sampling all documents once #
        pbar = tqdm(dataDUE.batchGenerate(batch_size=batch_size, keep_complete=True),
                    total = math.ceil(self.D_train * 1.0 / batch_size),
                    desc = '({0:^3})'.format(epoch))
        for i_batch, batch_size_real, data_batched in pbar:
            var_temp = self._fit_batchIntermediateInitialize()
            pars_topass = self._fit_single_epoch_pars_topass()

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

    def _fit_batchIntermediateInitialize(self):
        # instantiate vars every loop #
        vars = {
            "TI": np.zeros(self.K, dtype=np.float64),
            "Y1TF": np.zeros([self.K, self.A], dtype=np.float64),
            "Y0F" : np.zeros([self.A], dtype=np.float64),
            "UX": np.zeros([self.U, self.G], dtype=np.float64),
            "TV": np.zeros([self.K, self.V], dtype=np.float64),
            "FXE": np.zeros([self.A, self.G, self.E], dtype=np.float64),
            "YI": np.zeros([2], dtype=np.float64)
        }
        return vars

    def _fit_single_batch_cumulate(self, returned_fit_single_document, var_temp):
        # ends = []
        # ends.append(datetime.now())###
        # print "#### start _fit_single_batch_cumulate ####"

        d, doc_z, doc_f, doc_u, doc_x, doc_y, doc_TV, doc_FXE, doc_Y1TF = returned_fit_single_document  # parse returned from self._fit_single_document

        # ends.append(datetime.now())###

        # update document-level topic #
        self.z[d, :] = doc_z[:]
        self.f[d, :] = doc_f[:]

        var_temp["TI"] += doc_z

        # ends.append(datetime.now())###

        var_temp["Y1TF"] += doc_Y1TF

        # ends.append(datetime.now()) ###

        var_temp["Y0F"] += doc_f * doc_y[0]

        var_temp["UX"][doc_u, :] += doc_x

        # ends.append(datetime.now()) ###

        var_temp["TV"] += doc_TV

        # ends.append(datetime.now()) ###

        var_temp["FXE"] += doc_FXE

        var_temp["YI"] += doc_y

        # print "#### _fit_single_batch_cumulate detail profile: read, z, YI, Y0V, UX, Y1TV, TXE: ", [(ends[i]-ends[i-1]).total_seconds() for i in range(1, len(ends))]###

        return var_temp

    def _fit_single_batch_global_update(self, var_temp, batch_size_real, epoch):
        # ends = []###
        # ends.append(datetime.now())    ###

        lr = self._lrCal(epoch)

        # ends.append(datetime.now())###

        batch_weight = self.D_train * 1.0 / batch_size_real
        new_theta_temp = self.alpha + batch_weight * var_temp["TI"]
        new_pi_temp = self.delta + batch_weight * var_temp["Y1TF"]
        new_piB_temp = self.delta + batch_weight * var_temp["Y0F"]
        new_phi_temp = self.beta + batch_weight * var_temp["TV"]
        new_psi_temp = self.zeta + batch_weight * var_temp["UX"]
        new_eta_temp = self.gamma + batch_weight * var_temp["FXE"]
        new_c_temp = self.h + batch_weight * var_temp["YI"]

        # ends.append(datetime.now())###

        self.theta.update(new_theta_temp, lr)

        # ends.append(datetime.now())###

        self.pi.update(new_pi_temp, lr)

        self.piB.update(new_piB_temp, lr)

        # ends.append(datetime.now())###

        self.phi.update(new_phi_temp, lr)

        # ends.append(datetime.now())###

        self.psi.update(new_psi_temp, lr)

        # ends.append(datetime.now())###

        self.eta.update(new_eta_temp, lr)

        # ends.append(datetime.now())###

        self.c.update(new_c_temp, lr)

        # print "_fit_single_batch_global_update, detail profile for ## lr, add, theta, pi, phiB, phiT, psi, eta", [(ends[i] - ends[i-1]).total_seconds() for i in range(1, len(ends))]

    def _lrCal(self, epoch):
        ## rather than using epoch, using lr_step ##
        lr = float(self.lr["init"] * np.power((self.lr["tau"] + self.lr_step), - self.lr["kappa"]))
        self.lr_step += 1
        return lr

    def _ppl_compare(self, ppl_best, ppl):
        N_ppl = len(ppl)
        new_best = [ppl_best[i] for i in range(N_ppl)]
        best_flag = [False for i in range(N_ppl)]
        for i in range(N_ppl):
            if ppl_best[i] is None:
                # no such valid data to calculate #
                continue
            if ppl_best[i] > ppl[i]:
                new_best[i] = ppl[i]
                best_flag[i] = True
        return new_best, best_flag

    def _saveCheckPoint(self, epoch, ppl = None, filename = None):
        start = datetime.now()

        if filename is None:
            filename = self.checkpoint_file
        state = {
            "theta": self.theta.data,
            "pi": self.pi.data,
            "piB": self.piB.data,
            "eta": self.eta.data,
            "phi": self.phi.data,
            "psi": self.psi.data,
            "c": self.c.data,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "zeta": self.zeta,
            "z": self.z,
            "f": self.f,
            "epoch": epoch,
            "lr_step": self.lr_step,
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
        self.piB.initialize(new_data=state["piB"])
        self.eta.initialize(new_data=state["eta"])
        self.phi.initialize(new_data=state["phi"])
        self.psi.initialize(new_data=state["psi"])
        self.c.initialize(new_data=state["c"])
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.gamma = state["gamma"]
        self.delta = state["delta"]
        self.zeta = state["zeta"]
        self.z = state["z"]
        self.f = state["f"]
        self.epoch_init = state["epoch"] + 1
        try:
            self.lr_step = state["lr_step"]
        except KeyError as e:
            print "earlier version with no lr_step"
            self.lr_step = state["epoch"]
        ppl = state["ppl"]
        print "restore state from file '%s' on epoch %d with ppl: %s" % (filename, state["epoch"], str(ppl))

        duration = datetime.now() - start
        print "_restoreCheckPoint takes %f s" % duration.total_seconds()

        # for model display #
        self._estimateGlobal()

    def _log(self, string):
        with open(self.log_file, "a") as logf:
            logf.write(string.rstrip("\n") + "\n")

if __name__ == "__main__":
    model = MRT_SVI(2,1)
    ans = vars(model)
    print type(ans)
    print ans

