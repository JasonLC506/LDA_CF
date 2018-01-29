import numpy as np
import cPickle
from matplotlib import pyplot as plt

## models ##
from MRT_SVI import MRT_SVI
from MRT_SVI_YF import MRT_SVI_YF

from dataDUE_generator import dataDUELoader

EMOTICON_LIST = ["LIKE", "LOVE", "SAD", "WOW", "HAHA", "ANGRY"]


data_dir = "data/CNN_foxnews/"
data_prefix = "_CNN_foxnews_combined_K10"

id_map_file = data_dir + "id_map" + data_prefix
postcontent_dataW_file = data_dir + "dataW" + data_prefix
postcontent_dataToken_file = data_dir + "dataToken" + data_prefix
word_dictionary_file = data_dir + "word_dictionary" + data_prefix

id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))
dataW = cPickle.load(open(postcontent_dataW_file, "r"))
dataToken = cPickle.load(open(postcontent_dataToken_file, "r"))
word_dictionary = cPickle.load(open(word_dictionary_file, "r"))

# batch_rBp_dir = data_dir + "K10_batch_train/"
# batch_valid_on_shell_dir = data_dir + "K10_batch_on_shell/valid/"
# batch_valid_off_shell_dir = data_dir + "K10_batch_off_shell/valid/"
# batch_test_on_shell_dir = data_dir + "K10_batch_on_shell/test/"
# batch_test_off_shell_dir = data_dir + "K10_batch_off_shell/test/"
#
# meta_data_train_file = data_dir + "meta_data_train" + data_prefix
# meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid" + data_prefix
# meta_data_off_test_file = data_dir + "meta_data_off_shell_test" + data_prefix

#### period_foxnews data ####
# data_dir = "data/period_foxnews/"
# data_prefix = "_period_foxnews_K10"
#
# batch_rBp_dir = data_dir + "period_foxnews_K10_batch_train/"
# batch_valid_on_shell_dir = data_dir + "period_foxnews_K10_batch_on_shell/valid/"
# batch_valid_off_shell_dir = data_dir + "period_foxnews_K10_batch_off_shell/valid/"
# batch_test_on_shell_dir = data_dir + "period_foxnews_K10_batch_on_shell/test/"
# batch_test_off_shell_dir = data_dir + "period_foxnews_K10_batch_off_shell/test/"
#
# meta_data_train_file = data_dir + "meta_data_train" + data_prefix
# meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid" + data_prefix
# meta_data_off_test_file = data_dir + "meta_data_off_shell_test" + data_prefix
# meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid" + data_prefix
# meta_data_on_test_file = data_dir + "meta_data_on_shell_test" + data_prefix

#### period_foxnews_nolike data ####
data_dir = "data/period_foxnews_nolike/"

batch_rBp_dir = data_dir + "train/"
batch_valid_on_shell_dir = data_dir + "on_shell/valid/"
batch_valid_off_shell_dir = data_dir + "off_shell/valid/"
batch_test_on_shell_dir = data_dir + "on_shell/test/"
batch_test_off_shell_dir = data_dir + "off_shell/test/"

meta_data_train_file = data_dir + "meta_data_train"
meta_data_off_valid_file = data_dir + "meta_data_off_shell_valid"
meta_data_off_test_file = data_dir + "meta_data_off_shell_test"
meta_data_on_valid_file = data_dir + "meta_data_on_shell_valid"
meta_data_on_test_file = data_dir + "meta_data_on_shell_test"


def training(dataW, batch_rBp_dir, batch_valid_on_shell_dir=None, batch_valid_off_shell_dir=None, dataToken=None,
             dataDUE_loader=dataDUELoader, Model=MRT_SVI, hyperparameters = [], id_map_reverse = id_map_reverse, resume=None,
             batch_size=1024, lr_init=1.0, lr_kappa=0.1, random_shuffle=True,
             beta = 0.01, gamma=1.0, delta=0.01, zeta=0.1, converge_threshold_inner=0.01):
    K, G, A = hyperparameters
    model = Model(K, G, A)
    dataDUE = dataDUE_loader(meta_data_file=meta_data_train_file, batch_data_dir=batch_rBp_dir, dataToken=dataToken, id_map=id_map_reverse,
                             random_shuffle=random_shuffle)
    if batch_valid_on_shell_dir is not None:
        dataDUE_valid_on_shell = dataDUE_loader(meta_data_file=meta_data_train_file, batch_data_dir=batch_valid_on_shell_dir, dataToken=dataToken, id_map=id_map_reverse,
                                                random_shuffle=False)
    else:
        dataDUE_valid_on_shell = None
    if batch_valid_off_shell_dir is not None:
        dataDUE_valid_off_shell = dataDUE_loader(meta_data_file=meta_data_off_valid_file, batch_data_dir=batch_valid_off_shell_dir, dataToken=dataToken, id_map=id_map_reverse,
                                                 random_shuffle=False)
    else:
        dataDUE_valid_off_shell = None

    model.log_file += "K%d_G%d_A%d_batch%d_kappa_%f_beta%f_gamma%f_delta%f_zeta%f" % (K, G, A, batch_size, lr_kappa, beta, gamma, delta, zeta)
    model.checkpoint_file += "K%d_G%d_A%d_batch%d_kappa_%f_beta%f_gamma%f_delta%f_zeta%f" % (K, G, A, batch_size, lr_kappa, beta, gamma, delta, zeta)
    model.fit(dataDUE, resume= resume,
              batch_size=batch_size, lr_init=lr_init, lr_kappa=lr_kappa,
              beta=beta, gamma=gamma, delta=delta, zeta=zeta, converge_threshold_inner=converge_threshold_inner,
              dataDUE_valid_on_shell=dataDUE_valid_on_shell, dataDUE_valid_off_shell=dataDUE_valid_off_shell)


def modelDisplay(word_dictionary, Model=MRT_SVI, hyperparameters = [], resume=None):
    K, G, A = hyperparameters
    model = Model(K, G, A)
    model._restoreCheckPoint(resume)

    ## extract paras #
    theta = model.GLV["theta"]
    pi = model.GLV["pi"]
    eta = model.GLV["eta"]
    psi = model.GLV["psi"]
    phi = model.GLV["phi"]

    # pi = pi - np.mean(pi, axis=0, keepdims=True)


    # find top words for each topic #
    n_top_words = 8
    for i, topic_dist in enumerate(phi.tolist()):
        topic_words = np.array(word_dictionary)[np.argsort(topic_dist)][:-n_top_words:-1]
        print "Topic {}: {}".format(i, ','.join(topic_words))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # corpus-level topic distribution #
    ax1.plot(theta)
    ax1.set_title("corpus-level topic distribution")
    # ax1.legend()
    # topic-background distribution #
    # labels_B = ["Background", "topic"]
    for k in range(K):
        ax2.plot(pi[k], label="topic %d" % k)
    ax2.set_title("topic-attitude distribution")
    ax2.legend()
    # user-group distribution cumulative #
    ax3.pie(np.mean(psi, axis=0), labels=["group %d" % g for g in range(G)], autopct="%1.1f%%")
    ax3.set_title("mean user-group distribution")
    plt.legend()
    plt.show()

    f, axes = plt.subplots(1, G)
    # topic-emotion distribution for each group #
    for g in range(G):
        for a in range(A):
            axes[g].plot(eta[a, g, :], label="attitude %d" % a)
        axes[g].set_title("attitude-emotion distribution for group %d" % g)
    plt.legend()
    plt.show()

    TXE = np.tensordot(pi, eta, axes=(1,0))
    f, axes = plt.subplots(1, G)
    # topic-emotion distribution for each group #
    for g in range(G):
        for k in range(K):
            axes[g].plot(TXE[k, g, :], label="topic %d" % k)
        axes[g].set_title("topic-emotion distribution for group %d" % g)
    plt.legend()
    plt.show()

    # for d in range(model.z.shape[0]):
    #     print "document %d" % d
    #     print model.z[d]
    #     print model.f[d]

def modelDisplay_YF(word_dictionary, Model=MRT_SVI_YF, hyperparameters = [], resume=None):
    K, G, A = hyperparameters
    model = Model(K, G, A)
    model._restoreCheckPoint(resume)

    ## extract paras #
    theta = model.GLV["theta"]
    pi = model.GLV["pi"]
    piB = model.GLV["piB"]
    eta = model.GLV["eta"]
    psi = model.GLV["psi"]
    phi = model.GLV["phi"]
    c = model.GLV["c"]

    # pi = pi - np.mean(pi, axis=0, keepdims=True)


    # find top words for each topic #
    n_top_words = 8
    for i, topic_dist in enumerate(phi.tolist()):
        topic_words = np.array(word_dictionary)[np.argsort(topic_dist)][:-n_top_words:-1]
        print "Topic {}: {}".format(i, ','.join(topic_words))

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    # corpus-level topic distribution #
    ax1.plot(theta)
    ax1.set_title("corpus-level topic distribution")
    # ax1.legend()
    # topic-background distribution #
    # labels_B = ["Background", "topic"]
    for k in range(K):
        ax2.plot(pi[k], label="topic %d" % k)
    ax2.plot(piB, label="topic Background")
    ax2.set_title("topic-attitude distribution")
    ax2.legend()
    # user-group distribution cumulative #
    ax3.pie(np.mean(psi, axis=0), labels=["group %d" % g for g in range(G)], autopct="%1.1f%%")
    ax3.set_title("mean user-group distribution")

    ax4.pie(c, labels=["background", "topical"], autopct="%1.1f%%")
    ax4.set_title("background vs topical for attitude")
    plt.legend()
    plt.show()

    f, axes = plt.subplots(1, G)
    # topic-emotion distribution for each group #
    for g in range(G):
        for a in range(A):
            axes[g].plot(eta[a, g, :], label="attitude %d" % a)
        axes[g].set_title("attitude-emotion distribution for group %d" % g)
    plt.legend()
    plt.show()

    TXE = np.tensordot(pi, eta, axes=(1,0)) * c[1] + np.tensordot(piB, eta, axes=(0,0)) * c[0]
    f, axes = plt.subplots(1, G)
    # topic-emotion distribution for each group #
    for g in range(G):
        for k in range(K):
            axes[g].plot(TXE[k, g, :], label="topic %d" % k)
        axes[g].set_title("topic-emotion distribution for group %d" % g)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    K = 10
    G = 4
    A = 10
    training(dataW, batch_rBp_dir, batch_valid_on_shell_dir = batch_valid_on_shell_dir, batch_valid_off_shell_dir=batch_test_off_shell_dir,
             dataToken=dataToken,
             Model=MRT_SVI_YF,
             batch_size = 10000, lr_init=1.0, lr_kappa=0.0,
             beta=0.01, gamma=1.0, zeta=0.1, delta=0.1, converge_threshold_inner=0.001,
             hyperparameters=[K, G, A],
             id_map_reverse = id_map_reverse,
             resume = None)
    # modelDisplay(word_dictionary, Model=MRT_SVI, hyperparameters=[K, G, A], resume="ckpt/MRT_SVIK10_G3_A10_batch27239_kappa_0.000000_beta0.010000_gamma1.000000_delta0.010000_zeta0.100000_best_ppl[4]")
    # modelDisplay_YF(word_dictionary, hyperparameters=[K,G,A], resume="ckpt/MRT_SVI_YFK10_G5_A10_batch10000_kappa_0.000000_beta0.010000_gamma1.000000_delta0.100000_zeta0.100000_best_ppl[4]")
    # print sum(map(len, dataToken))
