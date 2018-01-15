import re
import numpy as np
from scipy.sparse import csr_matrix
import cPickle
from matplotlib import pyplot as plt

from emotion_topic_model import ETM
from eToT import eToT
from TTM import TTM
import lda
from post_content_extract import post_content_extract
from data_processing import Corpus, dataProcessing
from functions import probNormalize

np.seterr(divide='raise', invalid='raise', over='raise')

EMOTICON_LIST = ["LIKE", "LOVE", "SAD", "WOW", "HAHA", "ANGRY"]

filename = "data/CNN"
pattern = re.compile(r'^5550296508_')
posts = post_content_extract(filename, pattern)

emotions = cPickle.load(open("data/CNN_post_emotion", "r"))
# cp = Corpus(posts.values())
# cp.preprocessing()
#
# dataW = cp.matrix
# dataToken = cp.corpus
# # Ndocs = 1000
# # V = 10000
# E = 6
# # word_dist = np.arange(V)*1.0/V/(V-1)*2.0
# # dataW = csr_matrix(np.random.multinomial(10, word_dist, Ndocs))          # synthesize corpus data
# dataE = np.random.multinomial(10, np.arange(E).astype(np.float32)/float(E*(E-1)*0.5), cp.Ndocs) / 10.0       # synthesize emotion data

cp, dataE, id_map = dataProcessing(posts, emotions, emotion_normalize=False)

print "######### data statistics ##########"
print "Ndocs", cp.Ndocs
print "Ntokens", cp.Ntokens
print "V", cp.matrix.shape[1]
print "E", dataE.shape[1]

dataW = cp.matrix
model = TTM(K=5)
# model.fit(dataE,dataW, resume=None)

# ## lda ##
# model = lda.LDA(n_topics = 10, n_iter = 1500, random_state=1)
# model.fit(dataW)
# topic_word = model.topic_word_
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(cp.words)[np.argsort(topic_dist)][:-(n_top_words):-1]
#     print "Topic {}: {}".format(i, ','.join(topic_words))
# doc_topic = model.doc_topic_
# corpus_level_topic = np.mean(doc_topic, axis = 0)
# plt.plot(corpus_level_topic, label="corpus-level topic distribution")
# plt.legend()
# plt.show()



###### ETM #######
# model._restoreCheckPoint(filename="ckpt/ETM_K20")
# theta, phi = model.theta, model.phi
# # find top words for each topic #
# n_top_words = 8
# for i, topic_dist in enumerate(phi.tolist()):
#     topic_words = np.array(cp.words)[np.argsort(topic_dist)][:-n_top_words:-1]
#     print "Topic {}: {}".format(i, ','.join(topic_words))
# for i in range(6):
#     plt.plot(theta[i],label="e: %s" % EMOTICON_LIST[i])
# plt.legend()
# plt.show()

# ###### eToT ######
# model._restoreCheckPoint(filename="ckpt/eToT_K20")
# theta, phi, eta = model.theta, model.phi, model.eta
# # find top words for each topic #
# n_top_words = 8
# for i, topic_dist in enumerate(phi.tolist()):
#     topic_words = np.array(cp.words)[np.argsort(topic_dist)][:-n_top_words:-1]
#     print "Topic {}: {}".format(i, ','.join(topic_words))
# K, E = eta.shape
# topic_emotion = probNormalize(eta)      # take mean of dirichlet distribution
# for k in range(K):
#     plt.plot(topic_emotion[k], label = "topic: #%02d" % k)
# plt.legend()
# plt.show()

###### TTM ######
model._restoreCheckPoint(filename = "ckpt/TTM")
theta, pi, eta, phiB, phiT = model.theta, model.pi, model.eta, model.phiB, model.phiT
# find top words for each topic #
n_top_words = 8
for i, topic_dist in enumerate(phiT.tolist()):
    topic_words = np.array(cp.words)[np.argsort(topic_dist)][:-n_top_words:-1]
    print "Topic {}: {}".format(i, ','.join(topic_words))
n_top_words_B = 2 * n_top_words
topic_words = np.array(cp.words)[np.argsort(phiB)][:-n_top_words_B:-1]
print "Topic {}: {}".format("B", ",".join(topic_words))

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
# corpus-level topic distribution #
ax1.plot(theta, label="corpus-level topic distribution")
ax1.legend()
# topic-emotion distribution #
for k in range(eta.shape[0]):
    ax2.plot(eta[k], label = "topic %d" % k)
ax2.set_title("topic-emotion distribution")
ax2.legend()
# topic-background distribution #
labels_B = ["Background", "topic"]
ax3.pie(pi, labels = labels_B, autopct='%1.1f%%')
ax3.set_title("topic-background distribution")
plt.legend()
plt.show()