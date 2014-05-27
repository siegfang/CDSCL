
#
# Author: siegfang
#

import numpy as np

import functools
import operator
from collections import defaultdict
from itertools import chain

from debug import timeit
from .. import bolt



def parse_line(line):
    tokens = [tf for tf in line.rstrip().split(' ')]
    s, label = tokens[-1].split(':')
    assert s == "#label#"
    return label, [t for t in tokens[:-1] if len(t) > 0]

def vectorize(tokens, voc):
    doc = [(voc[term], int(freq)) for term, freq in tokens
           if term in voc]
    doc = sorted(doc)
    return doc

def disjoint_voc(s_voc, t_voc):
    n = len(s_voc)
    m = len(t_voc)
    s_voc = dict(zip(s_voc, range(n)))
    t_voc = dict(zip(t_voc, range(n, n + m)))
    return s_voc, t_voc, len(s_voc) + len(t_voc)

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
                label, tokens = parse_line(line)
                tokens = {}.fromkeys(tokens).keys()
                for token in tokens:
                    fd[token] += 1
    voc = set([t for t, c in fd.iteritems() if c >= mindf])
    return voc

@timeit
def load(fname, voc, dim, maxlines=-1):
    """
    """
    instances = []
    labels = []
    with open(fname) as f:
        for i, line in enumerate(f):
            if maxlines != -1 and i >= maxlines:
                break
            label, tokens = parse_line(line)
            doc = vectorize(tokens, voc)
            x = np.array(doc, dtype=bolt.sparsedtype)
            norm = np.linalg.norm(x['f1'])
            if norm > 0.0:
                x['f1'] /= norm
            instances.append(x)
            labels.append(label)
    instances = bolt.io.fromlist(instances, np.object)
    labels = np.array(labels)
    classes = np.unique(labels)
    labels = np.searchsorted(classes, labels).astype(np.float32)
    if len(classes) == 2:
        labels[labels == 0] = -1
    return bolt.MemoryDataset(dim, instances, labels), classes

def autolabel(instances, auxtask):
    labels = np.ones((instances.shape[0],), dtype=np.float32)
    labels *= -1
    for i, x in enumerate(instances):
        indices = x['f0']
        for idx in auxtask:
            if idx in indices:
                labels[i] = 1
                break

    return labels
