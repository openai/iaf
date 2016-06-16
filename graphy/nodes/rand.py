import theano
import theano.tensor as T
import graphy as G
import math
import numpy as np
import collections

def RandomVariable(sample, logp, entr, **params):
    return G.Struct(sample=sample, logp=logp, entr=entr, **params)

# TODO: turn these random variables functions into constructors

'''
Bernoulli variable
'''
def bernoulli(p, sample=None):
    if sample is None:
        sample = G.rng.binomial(p=p, dtype=G.floatX)
    logp = - T.nnet.binary_crossentropy(p, sample).flatten(2).sum(axis=1)
    entr = - (p * T.log(p) + (1-p) * T.log(1-p)).flatten(2).sum(axis=1)
    return RandomVariable(sample, logp, entr, p=p)

'''
Categorical variable
p: matrix with probabilities (each row should sum to one)
'''
def categorical(p, sample=None):
    if sample is None:
        sample = G.rng.multinomial(pvals=p, dtype='int32').argmax(axis=1)
    logp = - T.nnet.categorical_crossentropy(p, sample.flatten())
    entr = - (p * T.log(p)).sum(axis=1)
    return G.Struct(**{'sample':sample,'logp':logp,'entr':entr,'p':p})
    return RandomVariable(sample, logp, entr, p=p)

'''
4D Categorical variable
ulogp4d: 4D tensor with unnormalized log-probabilities at 2nd dimension
1st dimension goes over datapoints
2nd dimension has size n_vars*n_categories
3rd and 4th dimensions are the spatial dimensions

sample: 4D tensor of integer values (each from 0 to n_categories-1)
'''
def categorical4d(ulogp4d, n_vars=3, n_classes=256, sample=None):    
    shape4d = ulogp4d.shape
    ulogp_2d = ulogp4d.reshape((shape4d[0],n_vars,n_classes,shape4d[2],shape4d[3]))
    ulogp_2d = ulogp_2d.dimshuffle(0,1,3,4,2)
    ulogp_2d = ulogp_2d.reshape((shape4d[0]*n_vars*shape4d[2]*shape4d[3],n_classes))
    p_2d = T.nnet.softmax(ulogp_2d)
    if sample is None:
        sample_1d = G.rng.multinomial(pvals=p_2d, dtype='int32').argmax(axis=1)
        sample = sample_1d.reshape((shape4d[0],n_vars,shape4d[2],shape4d[3]))
    logp = - T.nnet.categorical_crossentropy(p_2d, sample.flatten())
    logp = logp.reshape((shape4d[0],n_vars*shape4d[2]*shape4d[3])).sum(axis=1)
    entr = - (p_2d * T.log(p_2d)).sum(axis=1)
    return RandomVariable(sample, logp, entr, ulogp4d=ulogp4d)


'''
Uniform random variable
[a, b]: domain
b > a
'''
def uniform(a, b, sample=None):
    logp = None
    if sample is None:
        sample = G.rng_curand.uniform(size=a.shape, low=a, high=b, dtype=G.floatX)
        # Warning: logp incorrect of y is outside of scope
        logp = -T.log(b-a).flatten(2).sum(axis=1)
    entr = T.log(b-a).flatten(2).sum(axis=1)
    return RandomVariable(sample, logp, entr, a=a, b=b)

'''
Diagonal Gaussian variable
mean: mean
logvar: log-variance
'''
def gaussian_diag(mean, logvar, sample=None):
    eps = None
    if sample is None:
        eps = G.rng_curand.normal(size=mean.shape)
        sample = mean + T.exp(.5*logvar) * eps
    logps = -.5 * (T.log(2*math.pi) + logvar + (sample - mean)**2 / T.exp(logvar))
    logp = logps.flatten(2).sum(axis=1)
    entr = (.5 * (T.log(2 * math.pi) + 1 + logvar)).flatten(2).sum(axis=1)
    kl = lambda p_mean, p_logvar: (.5 * (p_logvar - logvar) + (T.exp(logvar) + (mean-p_mean)**2)/(2*T.exp(p_logvar)) - .5).flatten(2).sum(axis=1)
    return RandomVariable(sample, logp, entr, mean=mean, logvar=logvar, kl=kl, logps=logps, eps=eps)


'''
Full-covariance Gaussian using a cholesky factor
mean (2D tensor): mean
logvar (2D tensor): log-variance
chol (3D tensor): cholesky factor minus the diagonal (upper triangular, zeros on diagonal)
'''
def gaussian_chol(mean, logvar, chol, sample=None):
    if sample != None:
        raise Exception('Not implemented')
    diag = gaussian_diag(mean, logvar)
    mask = T.shape_padleft(T.triu(T.ones_like(chol[0]), 1))
    sample = diag.sample + T.batched_dot(diag.sample, chol * mask)
    return RandomVariable(sample, diag.logp, diag.entr, mean=mean, logvar=logvar)

'''
Diagonal Gaussian variables
n_batch: batchsize
y: output
n_y: fixed parameter indicating dimensionality of the Gaussian
'''
def gaussian_spherical(shape=None, sample=None):
    if sample is None:
        sample = G.rng_curand.normal(shape)
    if shape is None:
        assert sample != None
        shape = sample.shape
    logp = -.5 * (T.log(2*math.pi) + sample**2).flatten(2).sum(axis=1)
    entr = (1.*T.prod(shape[1:]).astype(G.floatX)) * T.ones((shape[0],), dtype=G.floatX) * G.sharedf(.5 * (np.log(2.*math.pi)+1.))
    return RandomVariable(sample, logp, entr, shape=shape)

'''
Diagonal Laplace variable
mean: mean
scale: scale
'''
def laplace_diag(mean, logscale, sample=None):
    scale = .5*T.exp(logscale)
    if sample is None:
        u = G.rng_curand.uniform(size=mean.shape) - .5
        sample = mean - scale * T.sgn(u) * T.log(1-2*abs(u))
    logp = (- logscale - abs(sample-mean) / scale).flatten(2).sum(axis=1)
    entr = (1 + logscale).flatten(2).sum(axis=1)
    return RandomVariable(sample, logp, entr, mean=mean, scale=scale)


'''
Logistic random variable
'''
def logistic(mean, logscale, sample=None):
    scale = T.exp(logscale)
    if sample is None:
        u = G.rng_curand.uniform(size=mean.shape)
        _y = T.log(-u/(u-1)) #inverse CDF of the logistic
        sample = mean + scale * _y
    else:
        _y = -(sample-mean)/scale
    _logp = -_y - logscale - 2*T.nnet.softplus(-_y)
    logp = _logp.flatten(2).sum(axis=1)
    entr = logscale.flatten(2)
    entr = entr.sum(axis=1) + 2. * entr.shape[1]
    return RandomVariable(sample, logp, entr, mean=mean, logscale=logscale, _logp=_logp)

'''
Rectified Logistic random variable
'''
def rectlogistic(mean, logscale, sample=None):
    if sample is None:
        sample = T.maximum(0, logistic(mean, logscale).sample)
    mass0 = 1./(1+T.exp(-mean/T.exp(logscale)))
    logp = ((sample<=0) * mass0 + (sample>0) * logistic(mean, logscale, sample)._logp).flatten(2).sum(axis=1)
    entr = "Not implemented"
    return RandomVariable(sample, logp, entr, mean=mean, logscale=logscale)


'''
Discretized Logistic variable
mean: mean
logscale: logscale
'''
def discretized_logistic(mean, logscale, binsize, sample=None):
    scale = T.exp(logscale)
    if sample is None:
        u = G.rng_curand.uniform(size=mean.shape)
        _y = T.log(-u/(u-1)) #inverse CDF of the logistic
        sample = mean + scale * _y #sample from the actual logistic
        sample = T.floor(sample/binsize)*binsize #discretize the sample
    _sample = (T.floor(sample/binsize)*binsize - mean)/scale
    logps = T.log( T.nnet.sigmoid(_sample + binsize/scale) - T.nnet.sigmoid(_sample) + 1e-7)
    logp = logps.flatten(2).sum(axis=1)
    #raise Exception()
    entr = logscale.flatten(2)
    entr = entr.sum(axis=1) + 2. * entr.shape[1].astype(G.floatX)
    return RandomVariable(sample, logp, entr, mean=mean, logscale=logscale, logps=logps)

'''
Discretized Gaussian variable
mean: mean
logscale: logscale
'''
def discretized_gaussian(mean, logvar, binsize, sample=None):
    scale = T.exp(.5*logvar)
    if sample is None:
        _y = G.rng_curand.normal(size=mean.shape)
        sample = mean + scale * _y #sample from the actual logistic
        sample = T.floor(sample/binsize)*binsize #discretize the sample
    _sample = (T.floor(sample/binsize)*binsize - mean)/scale
    def _erf(x):
        return T.erf(x/T.sqrt(2.))
    logp = T.log( _erf(_sample + binsize/scale) - _erf(_sample) + 1e-7) + T.log(.5)
    logp = logp.flatten(2).sum(axis=1)
    #raise Exception()
    entr = (.5 * (T.log(2 * math.pi) + 1 + logvar)).flatten(2).sum(axis=1)
    return RandomVariable(sample, logp, entr, mean=mean, logvar=logvar)


'''
Discretized Laplace variable
mean: mean
scale: scale
'''
def discretized_laplace(mean, logscale, binsize, sample=None):
    scale = .5*T.exp(logscale)
    if sample is None:
        u = G.rng_curand.uniform(size=mean.shape) - .5
        sample = mean - scale * T.sgn(u) * T.log(1-2*abs(u))
        sample = T.floor(sample/binsize)*binsize #discretize the sample
    
    d = .5*binsize
    def cdf(x):
        z = x-mean
        return .5 + .5 * T.sgn(z) * (1.-T.exp(-abs(z)/scale))
    def logmass1(x):
        # General method for probability mass, but numerically unstable for large |x-mean|/scale
        return T.log(cdf(x+d) - cdf(x-d) + 1e-7)
    def logmass2(x):
        # Only valid for |x-mean| >= d
        return -abs(x-mean)/scale + T.log(T.exp(d/scale)-T.exp(-d/scale)) - np.log(2.).astype(G.floatX) 
    def logmass_stable(x):
        switch = (abs(x-mean) < d)
        return switch * logmass1(x) + (1-switch) * logmass2(x)
    
    logp = logmass_stable(sample).flatten(2).sum(axis=1)
    entr = None #(1 + logscale).flatten(2).sum(axis=1)
    return RandomVariable(sample, logp, entr, mean=mean, scale=scale)

'''
Laplace
NOT CONVERTED YET
'''
def zero_centered_laplace(name, w={}):
    w[name+'_logscale'] = G.sharedf(0.)
    def logp(v, w):
        return -abs(v).sum()/T.exp(w[name+'_logscale']) - v.size.astype(G.floatX) * (T.log(2.) + w[name+'_logscale'])
    postup = lambda updates, w:updates
    return G.Struct(logp=logp, postup=postup, w=w)

'''
Diagonal Gaussian variable
mean: mean
logvar: log-variance
NOT CONVERTED YET
'''
def zero_centered_gaussian(name, w={}):
    w[name+'_logvar'] = G.sharedf(0.)
    def logp(v, w):
        logvar = w[name+'_logvar']*10
        return v.size.astype(G.floatX) * -.5 * (T.log(2.*math.pi) + logvar) - .5 * (v**2).sum() / T.exp(logvar)
    postup = lambda updates, w:updates
    return G.Struct(logp=logp, postup=postup, w=w)


'''
Gaussian Scale Mixture
NOT CONVERTED YET
'''
def gsm(name, k, w={}, logvar_minmax=16):
    w[name+'_weight'] = G.sharedf(np.zeros((k,)))
    w[name+'_logvar'] = G.sharedf(np.random.randn(k)*.1)
    def logp(v, w):
        mixtureweights = T.exp(w[name+'_weight'])
        mixtureweights /= mixtureweights.sum()
        logvar = logvar_minmax*w[name+'_logvar']
        var = T.exp(logvar)
        if k == 0:
            return 0.
        if k == 1:
            return -.5*(v**2).sum()/var[0] - v.size.astype(G.floatX) * (.5*T.log(2.*math.pi) + logvar[0])
        p = 0.
        for i in range(k):
            p += mixtureweights[i] * T.exp(-.5*v**2/var[i]) / T.sqrt(2.*math.pi*var[i])
        logp = T.log(p).sum()
        return logp
    
    def postup(updates, w):
        updates[w[name+'_logvar']] = T.clip(updates[w[name+'_logvar']], -1., 1.)
        return updates
     
    return G.Struct(logp=logp, postup=postup, w=w)






