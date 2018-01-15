import re
import nltk
from nltk.corpus import stopwords
# nltk.download("stopwords")
import snowballstemmer
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import cPickle

from post_content_extract import post_content_extract
from functions import *

STEMMER = snowballstemmer.stemmer('english')
STOPWORDS = set(stopwords.words('english'))


class Corpus(object):
    def __init__(self, document_list):
        """
        document_list: a list of documents as strings
        """
        # raw data #
        self.corpus_raw_string = document_list

        ## tokenize ##
        # tokenized corpus #
        self.corpus_raw = map(self._string2tokens, self.corpus_raw_string)
        ### test ###
        #         self.words_raw = self.word_list(self.corpus_raw)
        #############

        ## clean ##                                                    # apply self.corpusClean()
        # token clean in document #
        self.corpus_token_cleaned = None
        # document cleaned in corpus (remove illegal documents) #
        self.doc_id_map = {"raw2clean": [], "clean2raw": []}  # find corresponding documents before/after clean
        self.corpus_cleaned = None

        ## vocaburary & indexing ##                                    # apply self.voc_index()
        self.Nwords = 0
        self.Ntokens = 0
        self.Ndocs = 0
        self.words = []  # index2word
        self.vocaburary = {}  # word2index
        self.corpus = None

        ## corpus2matrix ##
        self.matrix = None

    def preprocessing(self):
        self.corpusClean()
        self.voc_index()
        self.corpus2matrix()

    def corpusClean(self):
        self.corpus_token_cleaned = []
        self.corpus_cleaned = []
        cnt_cleaned = 0
        for d in range(len(self.corpus_raw)):
            doc_raw = self.corpus_raw[d]
            # token clean #
            doc_token_cleaned = map(self._tokenClean, doc_raw)
            self.corpus_token_cleaned.append(doc_token_cleaned)
            # document clean #
            doc_doc_cleaned = self._docClean(doc_token_cleaned)
            # document empty judge #
            if len(doc_doc_cleaned) != 0:
                self.corpus_cleaned.append(doc_doc_cleaned)
                self.doc_id_map["raw2clean"].append(cnt_cleaned)
                self.doc_id_map["clean2raw"].append(d)
                cnt_cleaned += 1
            else:
                self.doc_id_map["raw2clean"].append(-1)  # document deleted

    def voc_index(self):
        # check clean #
        if self.corpus_cleaned is None:
            self.corpusClean()  # default using cleaned corpus
        # vocaburary #
        onedoc = sum(self.corpus_cleaned, [])  # ! time consuming !
        self.Ntokens = len(onedoc)
        self.words = sorted(set(onedoc))
        self.Nwords = len(self.words)
        for i in range(self.Nwords):
            self.vocaburary[self.words[i]] = i
        # indexing #
        self.corpus = map(self._index, self.corpus_cleaned)
        self.Ndocs = len(self.corpus)

    def corpus2matrix(self):
        # check indexing #
        if self.corpus is None:
            self.voc_index()
        self.matrix = csr_matrix((self.Ndocs, self.Nwords), dtype=np.int8)
        for i_doc in range(self.Ndocs):
            doc = self.corpus[i_doc]
            l = len(doc)
            one_v = np.ones(l, dtype=np.int8)
            doc_matrix = coo_matrix((one_v, (one_v * i_doc, np.array(doc))), shape=[self.Ndocs, self.Nwords]).tocsr()
            self.matrix += doc_matrix

    def _string2tokens(self, doc_string):
        return doc_string.split(" ")

    def _tokenClean(self, token):
        # take lower cases #
        token = token.lower()

        # remove url #
        url_contained = re.search(r'http|\.com|\.org|\.it', token)
        if url_contained is not None or (re.search(r'\.', token) and re.search(r'/', token)):
            token = "URL"  # distinct upper case label

        # aggregate numbers #
        number_contained = re.search(r'\d', token)
        if number_contained is not None:
            token = re.sub(r'[\d\(\)\.,]', '', token)
            token = "NUMBER" + token  # distinct upper case label

        # remove punctuations #
        token = token.strip(":(),.?!;\t\'\"<>*-{}[]_=~|+/& ")
        # remove specific patterns #
        pattern_filter = re.compile(r'\'s$')
        if pattern_filter.search(token) is not None:
            token = token[:-2]

        # stemming #                                          # !creating doubtful results!
        token = STEMMER.stemWord(token).encode("ascii", "ignore")
        return token

    def _docClean(self, document):
        # clean tokens in document #
        new_doc = []
        for token in document:
            token_f = self._tokenFilter(token)
            if token_f is None:
                continue
            if type(token_f) is str:
                new_doc.append(token_f)
                continue
            if type(token_f) is list:
                for tf in token_f:
                    new_doc.append(tf)
                continue
        return new_doc

    def _tokenFilter(self, token):
        if len(token) == 0 or token in STOPWORDS:
            ### test ###
            if token in STOPWORDS:
                #                 print "stopword detected:", token
                pass
            return None
        if re.search(r'\.\.\.', token) is not None:
            tokens = token.split("...")
            tokens = map(self._tokenClean, tokens)
            return tokens

        return token

    def _index(self, document):
        # indexing document #
        document_indexed = map(lambda x: self.vocaburary[x], document)
        return document_indexed

    def word_list(self, corpus):
        onedoc = sum(corpus, [])
        return set(onedoc)


def dataProcessing(posts, emotions, emotion_normalize=True):
    ## subtract posts content without emotions
    posts_new = {}
    for key in posts.keys():
        if key not in emotions:
            print "content without emotion", key
            continue
        posts_new[key] = posts[key]
    posts = posts_new
    ## dictionary to list ##
    content_list = []
    id_list = []
    for post_id in posts:
        id_list.append(post_id)
        content_list.append(posts[post_id])
    ## clean text content ##
    cp = Corpus(content_list)
    cp.preprocessing()
    ## id_map ##
    id_map = [id_list[cp.doc_id_map["clean2raw"][i_doc]] for i_doc in range(cp.Ndocs)]
    ## emotion_list #
    emotion_list = [emotions[id_map[i_doc]] for i_doc in range(cp.Ndocs)]
    if emotion_normalize:
        dataE = probNormalize(np.array(emotion_list, dtype=np.float64))
    else:
        dataE = np.array(emotion_list, dtype = np.int32)

    return cp, dataE, id_map

def dataProcessingNoEmotion(posts, posts_ids):
    ## subtract posts not in posts_ids ##
    posts_new = {}
    for key in posts:
        if key not in posts_ids:
            print "content without emoticon", key
            continue
        posts_new[key] = posts[key]
    posts = posts_new
    ## dictionary to list ##
    content_list = []
    id_list = []
    for post_id in posts:
        id_list.append(post_id)
        content_list.append(posts[post_id])
    ## clean text content ##
    cp = Corpus(content_list)
    cp.preprocessing()
    ## id_map ##
    id_map = [id_list[cp.doc_id_map["clean2raw"][i_doc]] for i_doc in range(cp.Ndocs)]
    id_map_reverse = {}
    for i in range(len(id_map)):
        id_map_reverse[id_map[i]] = i
    return cp, id_map, id_map_reverse

if __name__ == "__main__":
    filename = "data/CNN"
    pattern = re.compile(r'^5550296508_')
    posts = post_content_extract(filename, pattern)
    cp = Corpus(posts.values())
    cp.preprocessing()
    print "Nwords %d, Ntokens %d, Ndocs %d" % (cp.Nwords, cp.Ntokens, cp.Ndocs)
    cnt = 1
    for word in cp.words:
        print cnt, word
        cnt += 1
        if cnt > 30000:
            print "too many to show"
            break