import theano
import theano.tensor as T
import collections as C
import numpy as np

'''
Operations on dictionaries of numpy tensors and/or theano symbolic variables.
Keys are strings.

Functions prepended with 'np_' only work on Numpy arrays
Functions appended with 'T_' only work on Theano symbolic variables
'''

def size(d):
    result = 0
    for i in d:
        result += d[i].size
    return result

#== Row operators

def getRows(d, ifrom, ito):
    return {i: d[i][ifrom:ito] for i in d}

def getRowsFromIndices(d, rowIndices):
    return {i: d[i][rowIndices] for i in d}

# ds: multiple dictionaries
# result: cols ifrom to ito from 'ds' 
def getRows_multiple(ds, ifrom, ito):
    return [{i: d[i][ifrom:ito] for i in d} for d in ds]

#=== Shuffle along first dimension of dicts

def shuffle(d):
    n_rows = d.itervalues().next().shape[0]
    idx = np.arange(n_rows)
    import time
    np.random.shuffle(idx)
    for i in d:
        t0 = time.time()
        d[i] = d[i][idx]
        #print i, time.time()-t0
    
#== Clone operations

def shallowClone(d):
    return {i: d[i] for i in d}

def clone(d):
    result = {}
    for i in d:
        result[i] = d[i].copy()
    return result

def cloneShared(d):
    result = {}
    for i in d:
        result[i] = theano.shared(d[i].get_value())
    return result

def np_cloneZeros(d):
    result = {}
    for i in d:
        result[i] = np.zeros(d[i].shape)
    return result

def np_cloneOnesN(d):
    result = {}
    for i in d:
        result[i] = np.ones(d[i].shape)
    return result

def T_cloneZeros(d):
    result = {}
    for i in d:
        result[i] = T.zeros_like(d[i])
    return result

def T_cloneOnes(d):
    result = {}
    for i in d:
        result[i] = T.ones_like(d[i])
    return result

def Tshared_cloneZeros(d):
    result = {}
    for i in d:
        result[i] = theano.shared(d[i].get_value() * 0.)
    return result

#=== Shape operations

# Get shapes of elements of d as a dict    
def getShapes(d):
    shapes = {}
    for i in d:
        shapes[i] = d[i].shape
    return shapes

# Set shapes of elements of d
def setShapes(d, shapes):
    result = {}
    for i in d:
        result[i] = d[i].reshape(shapes[i])
    return result

#=== Ordering operations
def ordered(d):
    return C.OrderedDict(sorted(d.items()))

# converts normal dicts to ordered dicts, ordered by keys
def ordereddicts(ds):
    return [ordered(d) for d in ds]
def orderedvals(ds):
    vals = []
    for d in ds:
        vals += ordered(d).values()
    return vals


#=== Type operations

def astype(d, _type):
    return {i: d[i].astype(_type) for i in d}

#def type(d):
#    return {i: d[i].type for i in d}

#=== Get/set value

def get_value(d):
    return {i: d[i].get_value() for i in d}

def set_value(d, d2, complete=True):
    for i in d:
        if i not in d2:
            if complete: raise Exception()
            continue
        d[i].set_value(d2[i])
 
#=== Merging/combining of multiple dicts

# Flatten sequence of dicts into one dict
# Input can also be nested sequence of sequences
# by default raises when keys overlap
def flatten(ds, raiseOnDuplicateKeys=True):
    if isinstance(ds, dict): return ds
    assert (isinstance(ds, list) or isinstance(ds, tuple))
    result = {}
    for d in ds:
        if (isinstance(d, list) or isinstance(d, tuple)):
            # recursion
            d = flatten(d, raiseOnDuplicateKeys)
        assert isinstance(d, dict)
        if raiseOnDuplicateKeys and any(i in d.keys() for i in result.keys()):
            print d.keys()
            print result.keys()
            raise Exception("Keys overlap overlap")
        result.update(d)
    return result

#=== Gradients

# Return gradients of scalar 'y' w.r.t. elements of d
# 'd' is a dict, or list of dicts
def T_grad(y, d, **kwargs):
    if type(d) is list:
        d = ordereddicts(d)
        vals = orderedvals(d)
        g = T.grad(y, vals, **kwargs)
        g_list = []
        idx = 0
        for i in range(len(d)):
            g_list += [dict(zip(d[i].keys(), g[idx:idx+len(d[i].keys())]))]
            idx += len(d[i].keys())
        return g_list
    else:
        d = ordered(d)
        keys = d.keys()
        grads = T.grad(y, d.values(), **kwargs)
        g = {keys[i]: grads[i] for i in range(len(grads))}
        return g

#=== Printing

def p(d):
    for i in d: print i+'\n', d[i]

def np_pNorm(d):
    for i in d: print i, np.linalg.norm(d[i])

def norm(d):
    return {i: np.linalg.norm(d[i]) for i in d}

def pShape(d):
    for i in d: print i, d[i].shape

def np_hasNaN(d):
    result = False
    for i in d: result = result or np.isnan(d[i]).any()
    return result

#=== Saving/loading

# Save/Load ndict to compressed file
# (a gzipped tar file, i.e. .tar.gz)
# if addext=True, then '.ndict' will be appended to filename
def np_savez(d, filename, addext=True):
    import tarfile, os
    if addext:
        filename = filename + '.ndict.tar.gz'
    fname1 = 'arrays.npz'
    fname2 = 'names.txt'
    _d = ordered(d)
    # Write values (arrays)
    np.savez(filename+'.'+fname1, *_d.values())
    # Write keys (names of arrays)
    with open(filename+'.'+fname2, 'w') as thefile:
        for key in _d.keys(): thefile.write("%s\n" % key)
    # Write TAR file
    tar = tarfile.open(filename, "w:gz")
    for fname in [fname1, fname2]:
        tar.add(filename+'.'+fname, fname)
        os.remove(filename+'.'+fname)
    tar.close()

# Loads ndict from file written with savez
def np_loadz(filename):
    import tarfile
    with tarfile.open(filename, 'r:gz') as tar:
        members = tar.getmembers()
        arrays = np.load(tar.extractfile(members[0]))
        names = tar.extractfile(members[1]).readlines()
        result = {names[i][:-1]: arrays['arr_'+str(i)] for i in range(len(names))}
    return ordered(result)
    
def shared(d, dtype=theano.config.floatX): # @UndefinedVariable
    result = {}
    for i in d:
        result[i] = theano.shared(np.asarray(d[i], dtype))
    return result
