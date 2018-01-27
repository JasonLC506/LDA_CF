import numpy as np
from warnings import warn
from functions import *


def _fit_single_document(docdata, pars_topass, max_iter_inner=500):
    """
    alternative optimization for local parameters for single document
    :return: [d, doc_z, doc_YI, doc_Y0V, doc_u, doc_x, doc_Y1TV, doc_TXE]
    """
    # start = datetime.now()  ###

    d, docToken, [doc_u, doc_e] = docdata

    doc_f = probNormalize(np.random.random([pars_topass["A"]]))
    doc_y = probNormalize(np.random.random([2]))
    doc_Md = pars_topass["Md"][d]
    doc_x_old = np.zeros([doc_Md, pars_topass["G"]])
    doc_z_old = np.zeros(pars_topass["K"])
    doc_y_old = doc_y.copy()
    doc_f_old = doc_f.copy()

    for inner_iter in xrange(max_iter_inner):
        doc_x = _fit_single_document_x_update(doc_f, doc_u, doc_e, pars_topass)
        doc_z = _fit_single_document_z_update(doc_f, doc_y, docToken, pars_topass)
        doc_y = _fit_single_document_y_update(doc_f, doc_z, pars_topass)
        doc_f = _fit_single_document_f_update(doc_z, doc_x, doc_y, doc_e, pars_topass)

        doc_x_old, doc_f_old, doc_z_old, doc_y_old, converge_flag, diff = _fit_single_document_convergeCheck(
            doc_x, doc_f, doc_z, doc_y, doc_x_old, doc_f_old, doc_z_old, doc_y_old, pars_topass)

        if converge_flag:

            return _fit_single_document_return(d, doc_x, doc_f, doc_z, doc_y, docToken, doc_u, doc_e, pars_topass)
    warn("Runtime warning: %d document not converged after %d" % (d, max_iter_inner))
    return _fit_single_document_return(d, doc_x, doc_f, doc_z, doc_y, docToken, doc_u, doc_e, pars_topass)

def _fit_single_document_x_update(doc_f, doc_u, doc_e, pars_topass):
    doc_x_unnorm_log_u = pars_topass["psi"].bigamma_data[doc_u]
    doc_x_unnorm_log_e = np.transpose(np.tensordot(doc_f, pars_topass["eta"].bigamma_data, axes=(0,0)), axes=(1,0))[doc_e,:]
    doc_x_unnorm_log = doc_x_unnorm_log_u + doc_x_unnorm_log_e
    return probNormalizeLog(doc_x_unnorm_log)

def _fit_single_document_z_update(doc_f, doc_y, docToken, pars_topass):
    doc_z_unnorm_log_w = np.sum(pars_topass["phi"].bigamma_data[:, docToken], axis=1)
    doc_z_unnorm_log_f = np.tensordot(pars_topass["pi"].bigamma_data, doc_f, axes=(1,0)) * doc_y[1]
    doc_z_unnorm_log_theta = pars_topass["theta"].bigamma_data
    doc_z_unnorm_log = doc_z_unnorm_log_w + doc_z_unnorm_log_f + doc_z_unnorm_log_theta
    return probNormalizeLog(doc_z_unnorm_log)

def _fit_single_document_y_update(doc_f, doc_z, pars_topass):
    doc_y_unnorm_log_c = pars_topass["c"].bigamma_data
    doc_y_unnorm_log_f = np.zeros([2], dtype=np.float64)
    doc_y_unnorm_log_f[0] = np.tensordot(doc_f, pars_topass["piB"].bigamma_data, axes=(0,0))
    doc_y_unnorm_log_f[1] = np.inner(np.dot(doc_z, pars_topass["pi"].bigamma_data), doc_f)
    doc_y_unnorm_log = doc_y_unnorm_log_c + doc_y_unnorm_log_f
    return probNormalizeLog(doc_y_unnorm_log)

def _fit_single_document_f_update(doc_z, doc_x, doc_y, doc_e, pars_topass):
    doc_f_unnorm_log_pi = np.dot(doc_z, pars_topass["pi"].bigamma_data) * doc_y[1]
    doc_f_unnorm_log_piB = pars_topass["piB"].bigamma_data * doc_y[0]
    doc_f_unnorm_log_e = np.tensordot(doc_x, pars_topass["eta"].bigamma_data[:, :, doc_e], axes=([0,1],[-1,1]))
    doc_f_unnorm_log = doc_f_unnorm_log_pi + doc_f_unnorm_log_piB + doc_f_unnorm_log_e
    return probNormalizeLog(doc_f_unnorm_log)

def _fit_single_document_convergeCheck(doc_x, doc_f, doc_z, doc_y, doc_x_old, doc_f_old, doc_z_old, doc_y_old, pars_topass=None):
    """ simple square difference check"""
    doc_Md = doc_x.shape[0]
    diff_x = np.linalg.norm(doc_x - doc_x_old) / np.sqrt(doc_Md * pars_topass["G"])
    diff_f = np.linalg.norm(doc_f - doc_f_old) / np.sqrt(pars_topass["A"])
    diff_z = np.linalg.norm(doc_z - doc_z_old) / np.sqrt(pars_topass["K"])
    diff_y = np.linalg.norm(doc_y - doc_y_old) / np.sqrt(2.0)
    diff_total = diff_x + diff_f + diff_z + diff_y
    if diff_total < pars_topass["converge_threshold_inner"]:
        converge = True
    else:
        converge = False
    return doc_x, doc_f, doc_z, doc_y, converge, diff_total

def _fit_single_document_return(d, doc_x, doc_f, doc_z, doc_y, docToken, doc_u, doc_e, pars_topass):
    # start = datetime.now() ###

    docToken_onehot = np.zeros([pars_topass["V"]])
    for n in range(docToken.shape[0]):
        docToken_onehot[docToken[n]] += 1.0
    doc_e_onehot = np.zeros([doc_e.shape[0], pars_topass["E"]])
    doc_e_onehot[np.arange(doc_e.shape[0]), doc_e] = 1

    doc_TV = np.outer(doc_z, docToken_onehot)
    doc_FXE = np.repeat(np.repeat(doc_f[:, np.newaxis, np.newaxis], pars_topass["G"], axis=1), pars_topass["E"], axis=2)
    doc_FXE = np.multiply(doc_FXE, np.dot(np.transpose(doc_x), doc_e_onehot))
    doc_Y1TF = np.outer(doc_z, doc_f) * doc_y[1]


    # duration = (datetime.now() - start).total_seconds()  ###
    # print "_fit_single_document_return takes %fs" % duration ###

    return d, doc_z, doc_f, doc_u, doc_x, doc_y, doc_TV, doc_FXE, doc_Y1TF

