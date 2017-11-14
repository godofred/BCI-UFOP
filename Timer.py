import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def tic(self):
        self.tstart = time.time()

    def toc(self):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)
        return time.time() - self.tstart