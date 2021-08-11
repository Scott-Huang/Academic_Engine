from datetime import datetime
from time import time, ctime

file = open('LOG.txt', 'a+')

def log(query, selection):
    file.write('%s: %s--%s' % (str(ctime(time())), query, selection))
