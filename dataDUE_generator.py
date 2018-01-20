import numpy as np
# from functions import multinomial, probNormalize
import cPickle
from multiprocessing import Process, Queue
import os
from datetime import datetime
import random

class dataDUELoader(object):
    def __init__(self, meta_data_file, batch_data_dir, id_map, dataToken=None, max_qsize = 5000, random_shuffle=False):

        self.E = 0                                  # dimension of emotion
        self.U = 0                                  # number of users
        self.Md = None                              # count of document-level total emoticons List[D]
        self.D = 0                                  # number of documents
        self.data_dir = batch_data_dir                    # data_directory
        with open(meta_data_file, "r") as f:
            meta_data = cPickle.load(f)
            self.E = meta_data["E"]
            self.U = meta_data["U"]
            self.D = meta_data["D"]
            self.Md = [0 for i in range(self.D)]
            for pid in meta_data["Md"]:
                if pid not in id_map:
                    # print pid    ### test
                    continue

                self.Md[id_map[pid]] = meta_data["Md"][pid]

        self.id_map = id_map                        # id_map between post_id and document_id
        self.batch_data_dir = batch_data_dir

        self.dataToken = dataToken

        # for data generation, ! not implemented yet ! #
        # will modify self._dataBatchReader() #
        # self.batch_size = batch_size
        self.random_shuffle = random_shuffle

        # multiprocess data reader #
        self.data_queue = Queue(maxsize=max_qsize)          # data queue for multiprocess
        self.data_reader = Process(target=self._dataBatchReader, args=(self.data_queue,))
        self.data_reader.daemon = True              # daemon process, killed automatically when main process ended

        self.data_reader.start()                    # !!! start when instantiated !!!

    def _dataBatchReader(self, data_queue, timeout=10000):
        while True:
            file_list = os.listdir(self.batch_data_dir)
            if self.random_shuffle:
                random.shuffle(file_list)
            for fn in file_list:
                start = datetime.now()
                with open(os.path.join(self.batch_data_dir, fn), "r") as f:
                    posts = cPickle.load(f)
                duration = datetime.now() - start
                # print "_dataBatchReader: load %s takes %f s" % (fn, duration.total_seconds())
                post_id_list = posts.keys()
                if self.random_shuffle:
                    random.shuffle(post_id_list)
                for post_id in post_id_list:
                    if post_id not in self.id_map:
                        continue
                    document_id = self.id_map[post_id]
                    if self.dataToken is not None:
                        data_queue.put([document_id, np.array(self.dataToken[document_id], dtype=np.int64), map(np.array, posts[post_id])], block=True,
                                       timeout=timeout)  # set max waiting time
                    else:
                        data_queue.put([document_id, posts[post_id]], block=True, timeout=timeout)   # set max waiting time
                del posts

    def dataReaderTerminate(self):
        ## in case, to manually terminate self.data_reader ##
        self.data_reader.terminate()
        self.data_reader.join()
        if not self.data_reader.is_alive():
            print "self.data_reader terminated"
        else:
            raise RuntimeError("self.data_reader cannot be terminated")

    def batchGenerate(self, batch_size = 1, keep_complete=True):
        N_batch = self.D / batch_size
        incomplete_batch = False
        if self.D % batch_size != 0:
            N_batch += 1
            incomplete_batch = True
        for i_batch in xrange(N_batch):
            if i_batch == (N_batch - 1):
                if incomplete_batch:
                    batch_size_real = self.D % batch_size
                else:
                    batch_size_real = batch_size
            else:
                batch_size_real = batch_size
            if keep_complete:
                batch_size_real = batch_size # even using more than one epoch data, keep batch_size the same
            yield i_batch, batch_size_real, self.generateSingleBatch(batch_size_real)

    def generateSingleBatch(self, batch_size):
        for i_samp in xrange(batch_size):
            yield self.data_queue.get(block=True, timeout=100)

    def generate(self, timeout=100):
        # depends on inputs, yield [document_id, [[reader_id],[emoticon]]] #
        ### example ###
        # for d in range(self.D):
        #     yield d, [np.arange(self.Md[d]), multinomial(probNormalize(np.random.random(self.E)), self.Md[d])]
        ###############
        """
        iteratively generate data for one epoch
        :param timeout: patience for waiting data
        """
        for doc_cnt in range(self.D):
            yield self.data_queue.get(block=True, timeout=100)

    def generateSingleBatch_packed(self, batch_size):
        data_batched = []
        for i_samp in xrange(batch_size):
            data_batched.append(self.data_queue.get(block=True, timeout=100))
        return data_batched

if __name__ == "__main__":
    data_prefix = "CNN_K10_"
    batch_rBp_dir = "data/" + data_prefix + "reactionsByPost_batch"
    meta_data_file = "data/" + data_prefix + "meta_data"
    id_map_file = "data/" + data_prefix + "post_id_map"

    id_map, id_map_reverse = cPickle.load(open(id_map_file, "r"))

    dataDUE = dataDUELoader(meta_data_file=meta_data_file, batch_data_dir=batch_rBp_dir, id_map=id_map_reverse)



