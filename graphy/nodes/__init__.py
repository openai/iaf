import theano
import theano.tensor as T
import graphy as G
import numpy as np

import sys

N = sys.modules[__name__]

# hyperparams
l2norm = True
logscale = True
logscale_scale = 3.
init_stdev = .1
maxweight = 3.

# Func for initializing parameters with random orthogonal matrix
def randorth(shape):
    from scipy.linalg import sqrtm, inv
    assert len(shape) == 2
    w = np.random.normal(0, size=shape)
    w = w.dot(inv(sqrtm(w.T.dot(w))))
    return G.sharedf(w)

# Softmax function
def softmax(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True)) 
    return e_x / e_x.sum(axis=1, keepdims=True)

def to_one_hot(x, n_y):
    # TODO: Replace this with built-in Theano function in extra_ops
    assert type(n_y) == int
    return T.eye(n_y)[x]
    
def dropout(x, p):
    if p > 0.:
        retain_p = 1-p
        x *= G.rng.binomial(x.shape, p=retain_p, dtype=G.floatX) / retain_p
    return x

# dropout where variable is replaced by noise with same marginal as input
def dropout_bernoulli(x, p):
    if p > 0:
        mask = G.rng.binomial(x.shape, p=p, dtype=G.floatX)
        p_noise = T.mean(x, axis=0, keepdims=True)
        noise = G.rng.binomial(x.shape, p=p_noise, dtype=G.floatX)
        x = (mask < .5) * x + (mask > .5) * noise
    return x

# Linear layer
def linear_l2(name, n_in, n_out, w):
    
    # L2 normalization of weights
    def l2normalize(_w):
        targetnorm=1.
        norm = T.sqrt((_w**2).sum(axis=0, keepdims=True))
        return _w * (targetnorm / norm)
    def maxconstraint(_w):
        return _w * (maxweight / T.maximum(maxweight, abs(_w).max(axis=0, keepdims=True)))
    
    w[name+'_w'] = G.sharedf(0.05*np.random.randn(n_in,n_out))
    
    if maxweight > 0:
        w[name+'_w'].set_value(maxconstraint(w[name+'_w']).tag.test_value)
    w[name+'_b'] = G.sharedf(np.zeros((n_out,)))
    if l2norm:
        if logscale:
            w[name+'_s'] = G.sharedf(np.zeros((n_out,)))
        else:
            w[name+'_s'] = G.sharedf(np.ones((n_out,)))
    else:
        print 'WARNING: constant rescale, these weights arent saved'
        constant_rescale = G.sharedf(np.zeros((n_out,)))
    
    
    def f(h, w):
        _w = w[name+'_w']
        if l2norm:
            _w = l2normalize(_w)
        h = T.dot(h, _w)
        if l2norm:
            if logscale:
                h *= T.exp(logscale_scale*w[name+'_s'])
            else:
                h *= abs(w[name+'_s'])
        else:
            h *= T.exp(constant_rescale)
        h += w[name+'_b']
        
        if '__init' in w:
            # Std
            std = (1./init_stdev) * h.std(axis=0) + 1e-8
            if name+'_s' in w:
                if logscale:
                    w[name+'_s'].set_value(-T.log(std).tag.test_value/logscale_scale)
                else:
                    w[name+'_s'].set_value((1./std).tag.test_value)
            else:
                constant_rescale.set_value(-T.log(std).tag.test_value)
                #w[name+'_w'].set_value((_w / std.dimshuffle('x',0)).tag.test_value)
            
            h /= std.dimshuffle('x',0)
            
            # Mean
            mean = h.mean(axis=0)
            w[name+'_b'].set_value(-mean.tag.test_value)
            h -= mean.dimshuffle('x',0)
            
            #print name, abs(w[name+'_w']).get_value().mean(), w[name+'_w'].get_value().std(), w[name+'_w'].get_value().max()

        #print name, abs(h).max().tag.test_value, abs(h).min().tag.test_value
        #h = T.printing.Print(name)(h)
        
        return h
    
    # Post updates: normalize weights to unit L2 norm
    def postup(updates, w):
        if l2norm and maxweight>0:
            updates[w[name+'_w']] = maxconstraint(updates[w[name+'_w']])
        return updates
    
    return G.Struct(__call__=f, postup=postup, w=w)

# Mean-only batchnorm, and a bias unit
def batchnorm_meanonly(name, n_h, w={}):
    w[name+'_b'] = G.sharedf(np.zeros((n_h,)))
    def f(h, w):
        h -= h.mean(axis=(0,2,3), keepdims=True)
        h += w[name+'_b'].dimshuffle('x',0,'x','x')
        return h
    return G.Struct(__call__=f, w=w)

    
'''
Nonlinear functions
(including parameterized ones)
'''
def nonlinearity(name, which, shape=None, w={}):
    
    if which == 'prelu':
        w[name] = G.sharedf(np.zeros(shape))
    if which == 'pelu':
        w[name] = G.sharedf(np.zeros(shape))
    if which == 'softplus2':
        w[name] = G.sharedf(np.zeros(shape))
    if which == 'softplus_shiftscale':
        w[name+'_in_s'] = G.sharedf(np.zeros(shape))
        w[name+'_in_b'] = G.sharedf(np.zeros(shape))
    if which == 'linearsigmoid':
        w[name+'_a'] = G.sharedf(.5*np.ones(shape))
        w[name+'_b'] = G.sharedf(.5*np.ones(shape))
    if which == 'meanonlybatchnorm_softplus':
        assert type(shape) == int
        w[name+'_b'] = G.sharedf(np.zeros(shape))
    if which == 'meanonlybatchnorm_relu':
        assert type(shape) == int
        w[name+'_b'] = G.sharedf(np.zeros(shape))
    
    def f(h, w=None):
        if which == None or which == 'None':
            return h
        elif which == 'tanh':
            return T.tanh(h)
        elif which == 'softmax':
            return T.nnet.softmax(h)
        elif which == 'prelu':
            return w[name]*h*(h<0.) + h*(h>=0.)
        elif which == 'relu':
            return h*(h>=0.)
        elif which == 'shiftedrelu':
            return T.switch(h < -1., -1., h)
        elif which == 'leakyrelu':
            return 0.01 * h*(h<0.) + h*(h>=0.)
        elif which == 'elu':
            return T.switch(h < 0., T.exp(h)-1, h)
        elif which == 'softplus':
            return T.nnet.softplus(h)
        elif which == 'softplus_shiftscale':
            return T.nnet.softplus(T.exp(w[name+'_in_s']) * h + w[name+'_in_b'])
        elif which == 'softplus2':
            return T.nnet.softplus(h) - w[name] * T.nnet.softplus(-h)
        elif which == 'linearsigmoid':
            return w[name+'_a'] * h + w[name+'_b'] * T.nnet.sigmoid(h)
        elif which == 'meanonlybatchnorm_softplus':
            h -= h.mean(axis=(0,2,3), keepdims=True)
            h += w[name+'_b'].dimshuffle('x',0,'x','x')
            return T.nnet.softplus(h)
        elif which == 'meanonlybatchnorm_relu':
            h -= h.mean(axis=(0,2,3), keepdims=True)
            h += w[name+'_b'].dimshuffle('x',0,'x','x')
            return T.nnet.relu(h)
        else:
            raise Exception("Unrecognized nonlinearity: "+which)
        
        
    return G.Struct(__call__=f, w=w)


# n_in is an int
# n_h is a list of ints
# n_out is an int or list of ints
# nl_h: nonlinearity of hidden units
# nl_out: nonlinearity of output
def mlp_l2(name, n_in, n_h, n_out, nl_h, nl_out=None, nl_in=None, w={}):
    
    if not isinstance(n_out, list) and isinstance(n_out, int):
        n_out = [n_out]
    
    # parameters for input perturbation
    if nl_in != None:
        f_nl_in = N.nonlinearity(name+'_in_nl', nl_in, (n_in,), w)
    
    # parameters for hidden units
    nh = [n_in]+n_h
    linear_h = []
    f_nl_h = []
    for i in range(len(n_h)):
        s = name+'_'+str(i)
        linear_h.append(N.linear_l2(s, nh[i], nh[i+1], w))
        f_nl_h.append(N.nonlinearity(s+'_nl', nl_h, (nh[i+1],), w))
    
    # parameters for output
    f_nl_out = []
    linear_out = []
    for i in range(len(n_out)):
        s = name+'_out_'+str(i)
        linear_out.append(N.linear_l2(s, n_h[-1], n_out[i], w))
        f_nl_out.append(N.nonlinearity(s+'nl', nl_out, (n_out[i],), w))
    
    def f(h, w, return_hiddens=False):
        
        if nl_in != None:
            h = f_nl_in(h, w)
        
        hiddens = []
        for i in range(len(n_h)):
            h = linear_h[i](h, w)
            h = f_nl_h[i](h, w)
            hiddens.append(h)
        
        out = []
        for i in range(len(n_out)):
            _out = linear_out[i](h, w)
            _out = f_nl_out[i](_out, w)
            out.append(_out)
        
        if len(n_out) == 1: out = out[0]
        
        if return_hiddens:
            return hiddens, out
        
        return out
    
    def postup(updates, w):
        for l in linear_h: updates = l.postup(updates, w)
        for l in linear_out: updates = l.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

