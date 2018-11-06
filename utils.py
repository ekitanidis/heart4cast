import numpy as np
from itertools import groupby

def find_consec(data, size):
    """ Finds all groups of contiguous numbers of a given size in data. These groups may overlap.
        Returns a list of tuples, where each tuple is the pair of indices in data enclosing the group. 
    """
    # find groups of contiguous numbers of minimum size
    ipairs = []
    k = 0
    for key, group in groupby(enumerate(data), lambda x:x[0]-x[1]):
        elems = len(list(group))
        if elems >= size:
            ipairs.append((k, k + elems - 1))
        k+=elems
    # split larger groups into multiple overlapping groups of exact size
    spairs = ipairs.copy()
    for i in ipairs:
        if (i[1] - i[0] + 1) > size:
            s = [(i[0] + j, i[0] + j + size - 1) for j in range(0, (i[1] - i[0] + 1) - size + 1, 1)]
            spairs.remove(i)
            spairs.extend(s)
    return spairs
