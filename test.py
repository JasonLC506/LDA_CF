import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import cPickle
from datetime import datetime
import time
from multiprocessing import Process, Queue
import os

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

if __name__ == "__main__":
    data_dir = "data/reactionsByPost_K10"
    start = datetime.now()
    # readfile(data_dir)
    q = Queue(maxsize = 5000)
    p = Process(target=readfileBatch, args = (q,))
    # p2 = Process(target=do_someting, args = (q,))
    p.start()
    # p2.start()
    do_someting(q)
    p.join()
    # p2.join()
    end = datetime.now()
    print "takes %f seconds" % (end-start).total_seconds()

