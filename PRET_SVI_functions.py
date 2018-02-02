import numpy as np
from warnings import warn
from functions import *

# def _fit_single_processor_batch(data_batched, pars_topass):
#     var_temp = _fit_batchIntermediateInitialize(pars_topass)
#     for docdata in data_batched:
#         returned = _fit_single_document(docdata, pars_topass)
#         var_temp = _fit_single_batch_cumulate(returned, var_temp)
#     return var_temp
#
# def _fit_batchIntermediateInitialize(pars_topass):
#     # instantiate vars every loop #
#     vars = {
#         "TI": np.zeros(pars_topass["K"], dtype=np.float64),
#         "YI": np.zeros(2, dtype=np.float64),
#         "Y0V": np.zeros(pars_topass["V"], dtype=np.float64),
#         "UX": np.zeros([pars_topass["U"], pars_topass["G"]], dtype=np.float64),
#         "Y1TV": np.zeros([pars_topass["K"], pars_topass["V"]], dtype=np.float64),
#         "TXE": np.zeros([pars_topass["K"], pars_topass["G"], pars_topass["E"]], dtype=np.float64),
#         "D_batch": 0,                                                                               # batch documents count
#         # "z": np.zeros([pars_topass["D"], pars_topass["K"]], dtype=np.float64)   ## if not needed,
#     }
#     return vars
#
# def _fit_single_batch_cumulate(returned_fit_single_document, var_temp):
#     d, doc_z, doc_YI, doc_Y0V, doc_u, doc_x, doc_Y1TV, doc_TXE = returned_fit_single_document  # parse returned from self._fit_single_document
#
#     var_temp["D_batch"] += 1
#     # var_temp["z"][d, :] = doc_z[:]                                  # update document-level topic #
#     var_temp["TI"] += doc_z
#     var_temp["YI"] += doc_YI
#     var_temp["Y0V"] += doc_Y0V
#     # var_temp["UX"] += doc_UX    # too sparse
#     var_temp["UX"][doc_u, :] += doc_x
#     var_temp["Y1TV"] += doc_Y1TV
#     var_temp["TXE"] += doc_TXE
#     return var_temp

def _fit_single_document(docdata, pars_topass, max_iter_inner=500):
    """
    alternative optimization for local parameters for single document
    :return: [d, doc_z, doc_YI, doc_Y0V, doc_u, doc_x, doc_Y1TV, doc_TXE]
    """
    # start = datetime.now()  ###

    d, docToken, [doc_u, doc_e] = docdata

    # doc_z = pars_topass["z"][d].copy()
    doc_z = probNormalize(np.random.random([pars_topass["K"]]))
    doc_Nd = pars_topass["Nd"][d]
    doc_Md = pars_topass["Md"][d]
    doc_x_old = np.zeros([doc_Md, pars_topass["G"]])
    doc_y_old = np.zeros([doc_Nd, 2])
    doc_z_old = doc_z.copy()

    # duration = (datetime.now() - start).total_seconds()###
    # print "_fit_single_document read data takes %fs" % duration###

    # time_profile = [0.0] * 4###
    # end4 = datetime.now() ###

    for inner_iter in xrange(max_iter_inner):
        # print "doc_x", doc_x_old
        # print "doc_y", doc_y_old
        # print "doc_z", doc_z


        doc_x = _fit_single_document_x_update(doc_z, doc_u, doc_e, pars_topass)

        # end1 = datetime.now()###
        # time_profile[0] += (end1 - end4).total_seconds()###

        ### no background test ###
        # doc_y = _fit_single_document_y_update(doc_z, docToken, pars_topass)
        doc_y = np.zeros([docToken.shape[0],2], dtype=np.float64)
        doc_y[:, 1] = 1.0 # no background

        # end2 = datetime.now()###
        # time_profile[1] += (end2 - end1).total_seconds()###

        doc_z = _fit_single_document_z_update(doc_y, doc_x, docToken, doc_e, pars_topass)

        # end3 = datetime.now()###
        # time_profile[2] += (end3 - end2).total_seconds()###

        doc_x_old, doc_y_old, doc_z_old, converge_flag, diff = _fit_single_document_convergeCheck(
            doc_x, doc_y, doc_z, doc_x_old, doc_y_old, doc_z_old, pars_topass)

        # end4 = datetime.now()###
        # time_profile[3] += (end4 - end3).total_seconds()###
        # print "doc_x", doc_x
        # print "doc_y", doc_y
        # print "doc_z", doc_z

        if converge_flag:

            ## test ###
            # duration = (datetime.now() - start).total_seconds() ###
            # print "_fit_single_document for doc_Md: %d, inner_iter: %d, takes: %fs" % (doc_Md, inner_iter, duration)###
            # print "_fit_single_document detail profile: x, y, z, convergecheck takes %f, %f, %f, %f s" % tuple(time_profile)###

            return _fit_single_document_return(d, doc_x, doc_y, doc_z, docToken, doc_u, doc_e, pars_topass)
    warn("Runtime warning: %d document not converged after %d" % (d, max_iter_inner))
    return _fit_single_document_return(d, doc_x, doc_y, doc_z, docToken, doc_u, doc_e, pars_topass)

def _fit_single_document_x_update(doc_z, doc_u, doc_e, pars_topass):
    doc_x_unnorm_log_u = pars_topass["psi"].bigamma_data[doc_u]
    doc_x_unnorm_log_e = np.transpose(np.tensordot(doc_z, pars_topass["eta"].bigamma_data, axes=(0,0)), axes=(1,0))[doc_e,:]
    doc_x_unnorm_log = doc_x_unnorm_log_u + doc_x_unnorm_log_e
    return probNormalizeLog(doc_x_unnorm_log)

def _fit_single_document_y_update(doc_z, docToken, pars_topass):
    doc_y_unnorm_log = np.zeros([docToken.shape[0], 2])
    doc_y_unnorm_log[:, 0] = pars_topass["phiB"].bigamma_data[docToken]
    doc_y_unnorm_log[:, 1] = np.dot(doc_z, pars_topass["phiT"].bigamma_data[:, docToken])
    doc_y_unnorm_log += pars_topass["pi"].bigamma_data
    return probNormalizeLog(doc_y_unnorm_log)

def _fit_single_document_z_update(doc_y, doc_x, docToken, doc_e, pars_topass):
    doc_z_unnorm_log_w = np.dot(pars_topass["phiT"].bigamma_data[:, docToken], doc_y[:, 1])

    doc_z_unnorm_log_e = np.tensordot(pars_topass["eta"].bigamma_data[:, :, doc_e], doc_x, axes=([2,1],[0,1]))

    doc_z_unnorm_log_theta = pars_topass["theta"].bigamma_data

    ### test ###
    # print "for document with Md: %d, Nd: %d" % (docToken.shape[0], doc_e.shape[0])
    # print "docE %s" % (str(list_count(doc_e, pars_topass["E"])))
    # print "doc_z_unnorm_log_theta", doc_z_unnorm_log_theta
    # print "doc_z_unnorm_log_w", doc_z_unnorm_log_w
    # print "doc_z_unnorm_log_e", doc_z_unnorm_log_e
    ### test ###
    doc_z_unnorm_log = doc_z_unnorm_log_w + doc_z_unnorm_log_e + doc_z_unnorm_log_theta

    return probNormalizeLog(doc_z_unnorm_log)

def _fit_single_document_convergeCheck(doc_x, doc_y, doc_z, doc_x_old, doc_y_old, doc_z_old, pars_topass=None):
        """ simple square difference check"""
        doc_Md = doc_x.shape[0]
        doc_Nd = doc_y.shape[0]
        diff_x = np.linalg.norm(doc_x - doc_x_old) / np.sqrt(doc_Md * pars_topass["G"])
        if doc_Nd == 0:
            diff_y = 0
        else:
            diff_y = np.linalg.norm(doc_y - doc_y_old) / np.sqrt(doc_Nd * 2)
        diff_z = np.linalg.norm(doc_z - doc_z_old) / np.sqrt(pars_topass["K"])
        diff_total = diff_x + diff_y + diff_z
        if diff_total < pars_topass["converge_threshold_inner"]:
            converge = True
        else:
            converge = False
        return doc_x, doc_y, doc_z, converge, diff_total

def _fit_single_document_return(d, doc_x, doc_y, doc_z, docToken, doc_u, doc_e, pars_topass):
    # start = datetime.now() ###

    docToken_onehot = np.zeros([docToken.shape[0], pars_topass["V"]])
    docToken_onehot[np.arange(docToken.shape[0]), docToken] = 1
    doc_e_onehot = np.zeros([doc_e.shape[0], pars_topass["E"]])
    doc_e_onehot[np.arange(doc_e.shape[0]), doc_e] = 1

    doc_YI = np.sum(doc_y, axis=0)
    doc_Y0V = np.dot(doc_y[:,0], docToken_onehot)
    # doc_UX = np.zeros([pars_topass["U"], pars_topass["G"]])
    # doc_UX[doc_u, :] = doc_x
    doc_Y1TV = np.dot(np.outer(doc_z, doc_y[:, 1]), docToken_onehot)
    doc_TXE = np.repeat(np.repeat(doc_z[:, np.newaxis, np.newaxis], pars_topass["G"], axis=1), pars_topass["E"], axis=2)
    doc_TXE = np.multiply(doc_TXE, np.dot(np.transpose(doc_x), doc_e_onehot))

    # duration = (datetime.now() - start).total_seconds()  ###
    # print "_fit_single_document_return takes %fs" % duration ###

    return d, doc_z, doc_YI, doc_Y0V, doc_u, doc_x, doc_Y1TV, doc_TXE



# def _ppl_new_process(data_queue, N_doc, pars_topass, epoch):
#     start = datetime.now()
#     _log("start _ppl", pars_topass["log_file"])
#
#     ppl_w_log = 0
#     ppl_e_log = 0
#     ppl_log = 0
#     for doc_cnt in range(N_doc):
#         docdata = data_queue.get(timeout=100)
#         try:
#             doc_ppl_log = _ppl_log_single_document(docdata, pars_topass)
#         except FloatingPointError as e:
#             _log("encounting underflow problem, no need to continue", pars_topass["log_file"])
#             return np.nan, np.nan, np.nan
#         ppl_w_log += doc_ppl_log[0]
#         ppl_e_log += doc_ppl_log[1]
#         ppl_log += doc_ppl_log[2]
#     # normalize #
#     ppl_w_log /= (sum(pars_topass["Nd"]))
#     ppl_e_log /= (sum(pars_topass["Md"]))
#     ppl_log /= pars_topass["D"]
#
#     duration = (datetime.now() - start).total_seconds()
#     _log("_ppl takes %fs" % duration, pars_topass["log_file"])
#
#     _log("ppl for epoch %d: %s" % (epoch, str([ppl_w_log, ppl_e_log, ppl_log])), pars_topass["log_file"])
#
#
# def _ppl_log_single_document(docdata, pars_topass):            ### potential underflow problem
#     d, docToken, [doc_u, doc_e] = docdata
#     prob_w_kv = (pars_topass["GLV"]["phiT"] * pars_topass["GLV"]["pi"][1] + pars_topass["GLV"]["phiB"] * pars_topass["GLV"]["pi"][0])
#     ppl_w_k_log = -np.sum(np.log(prob_w_kv[:, docToken]), axis=1)
#     ppl_w_k_scaled, ppl_w_k_constant = expConstantIgnore(- ppl_w_k_log, constant_output=True) # (actual ppl^(-1))
#
#     prob_e_mk = np.dot(pars_topass["GLV"]["psi"][doc_u, :], pars_topass["GLV"]["eta"])
#     ppl_e_k_log = - np.sum(np.log(prob_e_mk[np.arange(doc_u.shape[0]), :, doc_e]), axis=0)
#     ppl_e_k_scaled, ppl_e_k_constant = expConstantIgnore(- ppl_e_k_log, constant_output=True) # (actual ppl^(-1))
#     prob_k = pars_topass["GLV"]["theta"]
#
#
#     # for emoti given words
#     prob_e_m =  probNormalize(np.tensordot(prob_e_mk, np.multiply(prob_k, ppl_w_k_scaled), axes=(1,0)))
#     ppl_e_log = - np.sum(np.log(prob_e_m[np.arange(doc_u.shape[0]), doc_e]))
#     # for words given emoti ! same prob_w for different n
#     prob_w = probNormalize(np.tensordot(prob_w_kv, np.multiply(prob_k, ppl_e_k_scaled), axes=(0,0)))
#     ppl_w_log = - np.sum(np.log(prob_w[docToken]))
#     # for both words & emoti
#     try:
#         ppl_log = - (np.log(np.inner(ppl_w_k_scaled, np.multiply(ppl_e_k_scaled, prob_k)))
#                      + ppl_w_k_constant + ppl_e_k_constant)
#     except FloatingPointError as e:
#         raise e
#     return ppl_w_log, ppl_e_log, ppl_log
#
# def _log(string, log_file):
#     with open(log_file, "a") as logf:
#         logf.write(string.rstrip("\n") + "\n")
    
    
### test ###
# def list_count(doc_e, E=6):
#     count = np.zeros([doc_e.shape[0], E])
#     count[np.arange(doc_e.shape[0]), doc_e] = 1
#     return np.sum(count, axis=0)

"""
def _fit_single_document(self, docdata, max_iter_inner=500):

    d, docToken, [doc_u, doc_e] = docdata
    doc_z = self.z[d].copy()
    doc_Nd = self.Nd[d]
    doc_Md = self.Md[d]
    doc_x_old = np.zeros([doc_Md, self.G])
    doc_y_old = np.zeros([doc_Nd, 2])
    doc_z_old = doc_z.copy()
    for inner_iter in xrange(max_iter_inner):
        doc_x = self._fit_single_document_x_update(doc_z, doc_u, doc_e)
        doc_y = self._fit_single_document_y_update(doc_z, docToken)
        doc_z = self._fit_single_document_z_update(doc_y, doc_x, docToken, doc_e)
        doc_x_old, doc_y_old, doc_z_old, converge_flag, diff = self._fit_single_document_convergeCheck(
            doc_x, doc_y, doc_z, doc_x_old, doc_y_old, doc_z_old, threshold=self.converge_threshold_inner)
        if converge_flag:
            return self._fit_single_document_return(d, doc_x, doc_y, doc_z, docToken, doc_u, doc_e)
    warnings.warn("Runtime warning: %d document not converged after %d" % (d, max_iter_inner))
    return self._fit_single_document_return(d, doc_x, doc_y, doc_z, docToken, doc_u, doc_e)

def _fit_single_document_return(self, d, doc_x, doc_y, doc_z, docToken, doc_u, doc_e):
    docToken_onehot = np.zeros([docToken.shape[0], self.V])
    docToken_onehot[np.arange(docToken.shape[0]): docToken] = 1
    doc_e_onehot = np.zeros([doc_e.shape[0], self.E])
    doc_e_onehot[np.arange(doc_e.shape[0]), doc_e] = 1

    doc_YI = np.sum(doc_y, axis=0)
    doc_Y0V = np.dot(doc_y[:,0], docToken_onehot)
    doc_UX = np.zeros([self.U, self.G])
    doc_UX[doc_u, :] = doc_x
    doc_Y1TV = np.dot(np.outer(doc_z, doc_y[:, 1]), docToken_onehot)
    doc_TXE = np.repeat(np.repeat(doc_z[:, np.newaxis, np.newaxis], self.G, axis=1), self.E, axis=2)
    doc_TXE = np.multiply(doc_TXE, np.dot(np.transpose(doc_x), doc_e_onehot))
    return d, doc_z, doc_YI, doc_Y0V, doc_UX, doc_Y1TV, doc_TXE

def _fit_single_document_x_update(self, doc_z, doc_u, doc_e):
    doc_x_unnorm_log = self.psi.bigamma_data[doc_u]
    doc_x_unnorm_log += np.transpose(np.tensordot(doc_z, self.eta.bigamma_data[:, :, doc_e], axes=(0,0)), axes=(1,0))
    return probNormalizeLog(doc_x_unnorm_log)

def _fit_single_document_y_update(self, doc_z, docToken):
    doc_y_unnorm_log = np.zeros([docToken.shape[0], 2])
    doc_y_unnorm_log[:, 0] = self.phiB.bigamma_data[docToken]
    doc_y_unnorm_log[:, 1] = np.dot(doc_z, self.phiT.bigamma_data[:, docToken])
    doc_y_unnorm_log += self.pi.bigamma_data
    return probNormalizeLog(doc_y_unnorm_log)

def _fit_single_document_z_update(self, doc_y, doc_x, docToken, doc_e):
    doc_z_unnorm_log = np.dot(self.phiT.bigamma_data[:, docToken], doc_y[:, 1])
    doc_z_unnorm_log += np.tensordot(self.eta.bigamma_data[:, :, doc_e], doc_x, axes=([2,1],[0,1]))
    doc_z_unnorm_log += self.theta.bigamma_data
    return probNormalizeLog(doc_z_unnorm_log)

def _fit_single_document_convergeCheck(self, doc_x, doc_y, doc_z, doc_x_old, doc_y_old, doc_z_old, threshold=0.01):

        doc_Md = doc_x.shape[0]
        doc_Nd = doc_y.shape[0]
        diff_x = np.linalg.norm(doc_x - doc_x_old) / np.sqrt(doc_Md * self.G)
        diff_y = np.linalg.norm(doc_y - doc_y_old) / np.sqrt(doc_Nd * 2)
        diff_z = np.linalg.norm(doc_z - doc_z_old) / np.sqrt(self.K)
        diff_total = diff_x + diff_y + diff_z
        if diff_total < threshold:
            converge = True
        else:
            converge = False
        return doc_x, doc_y, doc_z, converge, diff_total

"""