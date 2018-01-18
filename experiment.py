import numpy as np
import cPickle
from matplotlib import pyplot as plt

## models ##
from PRET import PRET
from PRET_SVI import PRET_SVI

from dataDUE_generator import dataDUELoader

EMOTICON_LIST = ["LIKE", "LOVE", "SAD", "WOW", "HAHA", "ANGRY"]

data_prefix = "CNN_K10_"
batch_rBp_dir = "data/" + data_prefix + "reactionsByPost_batch"
meta_data_file = "data/" + data_prefix + "meta_data"
id_map_file = "data/" + data_prefix + "post_id_map"
postcontent_dataW_file = "data/" + data_prefix + "postContent_dataW"
postcontent_dataToken_file = "data/" + data_prefix + "postContent_dataToken"
word_dictionary_file = "data/" + data_prefix + "word_dictionary"

id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))
dataW = cPickle.load(open(postcontent_dataW_file, "r"))
dataToken = cPickle.load(open(postcontent_dataToken_file, "r"))
word_dictionary = cPickle.load(open(word_dictionary_file, "r"))

def training(dataW, batch_rBp_dir, dataDUE_loader=dataDUELoader, dataToken=None, Model=PRET, hyperparameters = [], id_map_reverse = id_map_reverse, resume=None,
             batch_size=1, random_shuffle=False):
    K, G = hyperparameters
    model = Model(K, G)
    dataDUE = dataDUE_loader(meta_data_file=meta_data_file, batch_data_dir=batch_rBp_dir, dataToken=dataToken, id_map=id_map_reverse,
                             batch_size=batch_size, random_shuffle=random_shuffle)
    model.fit(dataDUE, dataW, corpus=dataToken, resume= resume)

def modelDisplay(word_dictionary, Model=PRET, hyperparameters = [], resume=None):
    K, G = hyperparameters
    model = Model(K, G)
    model._restoreCheckPoint(resume)
    ## extract paras #
    theta = model.theta
    pi = model.pi
    eta = model.eta
    psi = model.psi
    phiB = model.phiB
    phiT = model.phiT

    # find top words for each topic #
    n_top_words = 8
    for i, topic_dist in enumerate(phiT.tolist()):
        topic_words = np.array(word_dictionary)[np.argsort(topic_dist)][:-n_top_words:-1]
        print "Topic {}: {}".format(i, ','.join(topic_words))
    n_top_words_B = 2 * n_top_words
    topic_words = np.array(word_dictionary)[np.argsort(phiB)][:-n_top_words_B:-1]
    print "Topic {}: {}".format("B", ",".join(topic_words))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # corpus-level topic distribution #
    ax1.plot(theta)
    ax1.set_title("corpus-level topic distribution")
    # ax1.legend()
    # topic-background distribution #
    labels_B = ["Background", "topic"]
    ax2.pie(pi, labels=labels_B, autopct='%1.1f%%')
    ax2.set_title("topic-background distribution")
    # user-group distribution cumulative #
    ax3.plot(np.mean(psi, axis=0))
    ax3.set_title("mean user-group distribution")
    plt.legend()
    plt.show()

    f, axes = plt.subplots(1, G)
    # topic-emotion distribution for each group #
    for g in range(G):
        for k in range(K):
            axes[g].plot(eta[k, g, :], label="topic %d" % k)
        axes[g].set_title("topic-emotion distribution for group %d" % g)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    K = 10
    G = 3
    training(dataW, batch_rBp_dir,
             dataToken=dataToken,
             Model=PRET_SVI,
             hyperparameters=[K, G],
             id_map_reverse = id_map_reverse,
             resume = None)
    # modelDisplay(word_dictionary, hyperparameters=[K, G], resume="ckpt/PRET_K10_G3")

