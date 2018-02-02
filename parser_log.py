import numpy as np
import sys
import re

pattern_1 = re.compile(r'^before training')
pattern_2 = re.compile(r'^epoch:')

for K in [6,7,8,9,15,20,30,40,50]:
    file_name = "log/period_foxnews/eToT_K%d" % K
    ppls = []
    with open(file_name, "r") as f:
        for line in f:
            if pattern_1.match(line):
                records = line.split(", ")
                ppls.append(map(lambda x: float(x.strip("ppl:( )\n")), records[1:]))
            if pattern_2.match(line):
                records = line.split(", ")
                ppls.append(map(lambda x: float(x.strip("ppl:( )\n")), records[1:]))

    result = np.array(ppls)
    print "K: %d" % K
    print "on shell_valid_min", np.min(result[:,1])
    print "off shell valid min", np.min(result[:,4])

