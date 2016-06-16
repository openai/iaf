import numpy as np
import theano
import ndict
import math, time, sys
import graphy as G
from collections import OrderedDict

'''
NaN detection for theano functions
from: http://deeplearning.net/software/theano/tutorial/debug_faq.html
'''
def nan_detection_mode():
    def detect_nan(i, node, fn):
        for output in fn.outputs:
            if np.isnan(output[0]).any():
                print '*** NaN detected ***'
                theano.printing.debugprint(node)
                print 'Inputs : %s' % [input[0] for input in fn.inputs]
                print 'Outputs: %s' % [output[0] for output in fn.outputs]
                #break
                raise Exception()
    return theano.compile.MonitorMode(post_func=detect_nan) # @UndefinedVariable

default_function_mode = 'FAST_RUN'
#default_function_mode = nan_detection_mode()

'''
Graphy function
Wraps theano function, same API, except:
- input x is a dict or sequence of dicts, so no worries about ordering
  (as with regular theano functions)
- output y can be either a dict of Theano vars or a single Theano variable
- Supports lazy compilation
- Supports minibatches
- Checks whether input keys match at compile- and runtime
'''
def function(x, y, lazy=False, _debug=False, checknan='raise', **kwargs):
    # Default keyword arguments
    if not kwargs.has_key('on_unused_input'):
        kwargs['on_unused_input'] = 'warn'
    if not kwargs.has_key('mode'):
        kwargs['mode'] = default_function_mode
    # Order the input dict
    x = ndict.ordered(ndict.flatten(x))
    # Check the output dict
    return_single_y = False
    if not isinstance(y, dict):
        return_single_y = True
        y = {str(y): y}
    y = ndict.ordered(y)
    # Lazily compiled function (saves a lot of time)
    f = [None]
    def _compile(verbose=True):
        t0 = time.time()
        print 'Compiling... ',
        #print '[graphy] Compiling function '+str(x.keys())+' => '+str(y.keys())+' ...'
        sys.stdout.flush()
        f[0] = theano.function(x.values(), y.values(), **kwargs)
        print "%.2f" % (time.time()-t0), 's'
    if not lazy:
        _compile()
    # The function to be called
    def func(data, n_batch=0, randomorder=True, data_global={}):
        data = ndict.ordered(ndict.flatten(data))
        data_global = ndict.ordered(ndict.flatten(data_global))        
        # Check if keys of 'x' and 'inputs' match
        allkeys = (data.keys() + data_global.keys())
        for i in range(len(data)):
            if x.keys()[i] not in allkeys:
                raise Exception('Non-matching keys:'+str(allkeys)+' vs. '+str(x.keys()))
        # Compile function if not already done
        if f[0] == None:
            _compile()
        if n_batch <= 0:
            # Get results
            _data = data.copy()
            _data.update(data_global)
            inputs_ordered = ndict.orderedvals((_data,))
            _result = f[0](*inputs_ordered)
            # Put it in a dictionary with the corresponding keys
            result = {y.keys()[i]: _result[i] for i in range(len(y))}
        else:
            # Minibatch-based evaluation.
            # This assumes that input and output are tensors, and the first dimension iterates of datapoints
            n_tot = data.itervalues().next().shape[0]
            n_minibatches = int(math.ceil(1. * n_tot / n_batch))
            
            n_tile = 1
            if n_batch > n_tot:
                assert n_batch%n_tot == 0
                n_tile = n_batch/n_tot
                
            indices = np.tile(np.arange(n_tot),n_tile)
            if randomorder:
                np.random.shuffle(indices)
                adict = dict(zip(np.tile(np.arange(n_tot),n_tile),indices))
                indices_inverse = sorted(adict, key=adict.get)
            
            results = []
            for i in range(n_minibatches):
                data_minibatch = ndict.getRowsFromIndices(data, indices[i*n_batch:(i+1)*n_batch])
                data_minibatch.update(data_global)
                inputs_ordered = ndict.orderedvals((data_minibatch,))
                results.append(f[0](*inputs_ordered))
                if _debug:
                    print 'Function debug', i, results[-1]
                if checknan == 'raise':
                    if np.isnan(np.sum(results[-1])):
                        print results[-1]
                        raise Exception("NaN detected")
            result = {y.keys()[i]: np.concatenate([results[j][i] for j in range(n_minibatches)]) for i in range(len(y))}
            if randomorder:
                result = ndict.getRowsFromIndices(result, indices_inverse)
        
        result = OrderedDict(sorted(result.items()))
        
        # Return result
        #raise Exception()
        if return_single_y:
            return result[result.keys()[0]]
        return result
    # Return the func
    return G.Struct(__call__=func, f=f)

# f: a function (as above) that returns ndict
# function: a function that returns arguments for f
# concat_axis: axis over which to concatenate the results 
def loop(f, f_data, n_batch, n_its, concat_axis=0):
    assert n_its >= 1
    results = [f(f_data(), n_batch=n_batch) for i in range(n_its)]
    result = {i: np.concatenate([results[j][i] for j in range(n_its)], axis=0) for i in results[0].keys()}
    return result
