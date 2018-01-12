import re

def post_content_extract(filename, pattern = re.compile(r'^5550296508_')):
    ## pattern default for CNN posts ##
    f = open(filename, "r")
    posts = {}
    last_id = None
    content = []
    for line in f.readlines():
        if pattern.match(line) is not None:
            if last_id is None:
                pass
            else:
                # write for last_id #
                if last_id in posts:
                    print "duplicate id", last_id
                else:
                    posts[last_id] = " ".join(content[:-1])
            last_id = line.rstrip("\n")
            content = []
        else:
            content.append(line.rstrip("\n").decode('utf-8').encode("ascii", "ignore"))
    f.close()
    return posts