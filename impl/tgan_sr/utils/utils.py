from timeit import default_timer as timer
import math
import sys
from contextlib import contextmanager


@contextmanager
def nice_open(filename=None, mode='r'): # thanks to https://stackoverflow.com/questions/17602878/how-to-handle-both-with-open-and-sys-stdout-nicely
    if filename is None:
        res = None
        do_close = False
    elif filename == '-':
        res = sys.stdin if mode=='r' else sys.stdout
        do_close = False
    else:
        res = open(filename, mode)
        do_close = True
    try:
        yield res
    finally:
        if do_close:
            res.close()


def abbrev_count(count):
    log_count = math.floor(math.log10(count))
    k_exponent = math.floor(log_count / 3)
    suffixes = ['', 'k', 'm']
    return '{:g}{}'.format(count / 10**(k_exponent*3), suffixes[k_exponent])


def strfdelta_hms(delta):
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 60*60)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def increment(d, k, v=1):
    if k in d:
        d[k] += v
    else:
        d[k] = v