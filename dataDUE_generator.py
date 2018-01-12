import numpy as np
# from functions import multinomial, probNormalize
import cPickle
from multiprocessing import Process, Queue
import os
from datetime import datetime

class dataDUE(object):
    def __init__(self, meta_data_file, batch_data_dir, id_map, max_qsize = 5000, batch_size=1, random_shuffle=False):

        self.E = 0                                  # dimension of emotion
        self.U = 0                                  # number of users
        self.Md = None                              # count of document-level total emoticons List[D]
        self.D = 0                                  # number of documents
        self.data_dir = batch_data_dir                    # data_directory
        with open(meta_data_file, "r") as f:
            meta_data = cPickle.load(f)
            self.E = meta_data["E"]
            self.U = meta_data["U"]
            self.Md = meta_data["Md"]
            self.D = meta_data["D"]
        self.id_map = id_map                        # id_map between post_id and document_id
        self.batch_data_dir = batch_data_dir

        # for data generation, ! not implemented yet ! #
        # will modify self._dataBatchReader() #
        self.batch_size = batch_size
        self.random_shuffle = random_shuffle

        # multiprocess data reader #
        self.data_queue = Queue(maxsize=max_qsize)          # data queue for multiprocess
        self.data_reader = Process(target=self._dataBatchReader, args=(self.data_queue,))
        self.data_reader.daemon = True              # daemon process, killed automatically when main process ended

        self.data_reader.start()                    # !!! start when instantiated !!!

    def _dataBatchReader(self, data_queue, timeout=10000):
        while True:
            for fn in os.listdir(self.batch_data_dir):
                start = datetime.now()
                with open(os.path.join(self.batch_data_dir, fn), "r") as f:
                    posts = cPickle.load(f)
                duration = datetime.now() - start
                print "_dataBatchReader: load %s takes %f s" % (fn, duration.total_seconds())
                for post_id in posts:
                    document_id = self.id_map[post_id]
                    data_queue.put([document_id, posts[post_id]], block=True, timeout=timeout)   # set max waiting time

    def dataReaderTerminate(self):
        ## in case, to manually terminate self.data_reader ##
        self.data_reader.terminate()
        self.data_reader.join()
        if not self.data_reader.is_alive():
            print "self.data_reader terminated"
        else:
            raise RuntimeError("self.data_reader cannot be terminated")

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


