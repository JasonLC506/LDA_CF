import re
import numpy as np
from scipy.sparse import csr_matrix
import cPickle
from matplotlib import pyplot as plt

from post_content_extract import post_content_extract
from data_processing import Corpus, dataProcessingNoEmotion

filename = "data/CNN"
pattern = re.compile(r'^5550296508_')
posts = post_content_extract(filename, pattern)

## legal post with emoticons ##
post_ids_filename = "data/CNN_post_ids"
post_ids = cPickle.load(open(post_ids_filename, "r"))

cp, id_map, id_map_reverse = dataProcessingNoEmotion(posts, post_ids)

# indexed post_content #
with open("data/CNN_K10_postContent_dataW", "w") as f:
    cPickle.dump(cp.matrix, f)

# tokenized indexed post_content #
with open("data/CNN_K10_postContent_dataToken", "w") as f:
    cPickle.dump(cp.corpus, f)

# dictionary #
with open("data/CNN_K10_word_dictionary", "w") as f:
    cPickle.dump(cp.words, f)

# id_map & _reverse #
with open("data/CNN_K10_post_id_map", "w") as f:
    cPickle.dump([id_map, id_map_reverse], f)
