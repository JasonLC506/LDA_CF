import numpy as np
import cPickle
from matplotlib import pyplot as plt

## models ##
from PRET import PRET

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
    dataDUE = dataDUE_loader(meta_data_file=meta_data_file, batch_data_dir=batch_rBp_dir, id_map=id_map_reverse,
                             batch_size=batch_size, random_shuffle=random_shuffle)
    model.fit(dataDUE, dataW, corpus=dataToken, resume= resume)

if __name__ == "__main__":
    K = 10
    G = 3
    training(dataW, batch_rBp_dir, dataToken=dataToken,
             Model=PRET, hyperparameters=[K, G],
             id_map_reverse = id_map_reverse,
             resume = None)

