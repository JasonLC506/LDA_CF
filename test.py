import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import cPickle
from datetime import datetime
import time
from multiprocessing import Process, Queue, Pool, cpu_count
import os
from matplotlib import pyplot as plt
import itertools
from tqdm import tqdm
import random

from functions import *

def doSomeThing(q):
    obj = q.get()
    if obj is None:
        print "empty queue"
        return -1
    print len(obj)
    cnt = 0
    for i in range(100000):
        cnt += 1
    return 1

def readfile(data_dir):
    posts = {}

    cnt = 0
    for filename in os.listdir(data_dir):
        result = cPickle.load(open(os.path.join(data_dir, filename), "r"))
        posts[filename] = result

        cnt += 1
        if (cnt + 1) % 100 == 0:
            print cnt
        if (cnt + 1) % 5000 == 0:
            cPickle.dump(posts, open("data/batch_%d" % cnt, "w"))
            del posts
            posts = {}
    if len(posts)>0:
        cPickle.dump(posts, open("data/batch_%d" % cnt, "w"))
        del posts

batch_files = ["data/batch_%d" % batch for batch in [4999, 9999, 14999, 19999, 23362]]
def readfileBatch(q, batch_files = batch_files):
    cnt = 0
    for fn in batch_files:
        start = datetime.now()
        posts = cPickle.load(open(fn, "r"))
        duration = datetime.now() - start
        print "load %s takes %f s" % (fn, duration.total_seconds())
        for post_id in posts:
            q.put([post_id, posts[post_id]], block=True)
            cnt += 1
            if (cnt + 1) % 100 == 0:
                print cnt

count = 0
def do_someting(q):
    global count
    while True:
        cnt = 0
        for i in range(int(1e+7)):
            cnt += 1
        r = q.get(block=True)
        # print "get", r[0]
        count += 1
        if count % 100 == 0:
            print "done", count
        if r is None:
            print "main_ended"
            break

# if __name__ == "__main__":
#     data_dir = "data/reactionsByPost_K10"
#     start = datetime.now()
#     # readfile(data_dir)
#     q = Queue(maxsize = 5000)
#     p = Process(target=readfileBatch, args = (q,))
#     # p2 = Process(target=do_someting, args = (q,))
#     p.start()
#     # p2.start()
#     do_someting(q)
#     p.join()
#     # p2.join()
#     end = datetime.now()
#     print "takes %f seconds" % (end-start).total_seconds()


# data_dir = "data/CNN_reactionsByPost_K10"
# post_ids = set([])
# for post_id in os.listdir(data_dir):
#     post_ids.add(post_id)
# with open("data/CNN_post_ids", "w") as f:
#     cPickle.dump(post_ids, f)



# c = [np.arange(i+1) for i in range(10000)]

def multinomial_single(probability):
    rnd = np.random.random()
    cpd = 0.0
    for i in xrange(probability.shape[0]):
        cpd += probability[i]
        if rnd <= cpd:
            return i
    return None

def multinomial_single_unnorm(probability):
    rnd = np.random.random() * np.sum(probability)
    cpd = 0.0
    for i in xrange(probability.shape[0]):
        cpd += probability[i]
        if rnd <= cpd:
            return i
    return None

def f(data):
    x,b = data
    cnt = 0
    for i in range(10000000/x/x):
        cnt += np.sqrt(b)

    return x*x+b

# def batched_data(data_q, batch_size):
#     for i in xrange(batch_size):
#         yield data_q.get()
#
# def iteriter(N, batch_size, data_q):
#     N_batch = N / batch_size
#     if N % batch_size != 0:
#         N_batch += 1
#     for i_batch in xrange(N_batch):
#         if i_batch == N_batch-1:
#             batch_size_true = N % batch_size
#         else:
#             batch_size_true = batch_size
#         yield i_batch, batched_data(data_q, batch_size_true)
#
# if __name__ == "__main__":
#     q = Queue()
#     q.daemon = True
#     for i in range(10000):
#         q.put(i)
#     for i_batch, data_batched in iteriter(1000, 30, q):
#         print i_batch
#         for data_sample in data_batched:
#             print data_sample

# if __name__ == "__main__":
#     print cpu_count()
#     pool = Pool(processes=3)
#     # results = [pool.apply_async(f,(i,)) for i in range(1,10)]
#     start = datetime.now()
#     returned = pool.imap_unordered(f, xrange(1, 1000), chunksize=1)
#     results = []
#     for item in returned:
#         results.append(item)
#         # print item
#     print len(set([res for res in results]))
#     print (datetime.now() - start).total_seconds()

class A(object):
    def __init__(self):
        self.a = 0
        self.b = [1, 2]

    def varsChange(self):
        ans = vars(self)
        print type(ans), ans["b"], type(ans["b"])
        del ans["b"]

def objGenerate(N):
    for i in xrange(N):
        yield [i+1,2]

def iterGenerate(N):
    for i in xrange(N):
        yield objGenerate(N)

if __name__ == "__main__":
    c = [1,2,3]
    d = c[1:]
    print d
