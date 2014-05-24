
#
# Author: siegfang
#

import numpy as np

import functools
import operator
from collections import defaultdict
from itertools import chain

from debug import timeit
# from .externals import bolt


def parse_labeled_line(line):
    tokens = [tf for tf in line.rstrip().split(' ')]
    s, label = tokens[-1].split(':')
    assert s == "#label#"
    return label, [t for t in tokens if len(t) > 0]


def count(*datasets):
    """Counts the example frequency of each feature in a list
    of datasets. All data sets must have the same dimension.

    Returns
    -------
    counts : array, shape = [datasets[0].dim]
        counts[i] holds the number of examples in data sets for
        which the i-th feature is non-zero.
    """
    if len(datasets) > 1:
        assert functools.reduce(operator.eq, [ds.dim for ds in datasets])
    counts = np.zeros((datasets[0].dim,), dtype=np.uint32)  # FIXME here was uint16
    for x, y in chain(*datasets):
        counts[x["f0"]] += 1
    return counts

@timeit
def vocabulary(*bowfnames, **kargs):
    """
    it supports the following kargs:
    - mindf: min document frequency (default 2).
    - maxlines: maximum number of lines to read (default -1).
    """
    mindf = kargs.get("mindf", 2)
    maxlines = kargs.get("maxlines", -1)
    fd = defaultdict(int)
    for fname in bowfnames:
        with open(fname) as f:
            for i, line in enumerate(f):
                if maxlines != -1 and i >= maxlines:
                    break
                label, tokens = parse_labeled_line(line)
                for token, freq in tokens:
                    fd[token] += 1
    voc = set([t for t, c in fd.iteritems() if c >= mindf])
    return voc