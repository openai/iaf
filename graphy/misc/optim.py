import graphy as G
import theano
import theano.tensor as T
import math
import numpy as np

from collections import OrderedDict

def SGD(w, objective, alpha=.1):
    print 'SGD', 'alpha:',alpha
    g = T.grad(objective.sum(), w, disconnected_inputs='warn')
    updates = OrderedDict()
    for i in range(len(g)):
        updates[w[i]] = w[i] + alpha * g[i]
    return updates

# Adam
def Adam(ws, objective, alpha=.0003, beta=.9, gamma=.999):
    print 'Adam', 'alpha:',alpha,'beta1:',beta,'gamma:',gamma
    
    new = OrderedDict()

    gs = G.ndict.T_grad(objective.sum(), ws, disconnected_inputs='warn') #warn/raise
    
    it = G.sharedf(0.)
    new[it] = it + 1.
    
    fix1 = 1-beta**(it+1.)
    fix2 = 1-gamma**(it+1.) # To make estimates unbiased
    lr_t = alpha * T.sqrt(fix2) / fix1
    
    ws_avg = []
    for j in range(len(ws)):
        w_avg = {}
        for i in ws[j]:
            w = ws[j][i]
            g = gs[j][i]
            
            # Initial values
            shape = w.get_value().shape
            m = G.sharedf(np.zeros(shape))
            v = G.sharedf(np.zeros(shape))
            w_avg[i] = G.sharedf(np.zeros(shape))
            
            # Updates
            new[m] = beta * m + (1-beta) * g
            new[v] = gamma * v + (1-gamma) * g**2
            new[w] = w + lr_t * new[m] / (T.sqrt(new[v]) + 1e-8)
            new[w_avg[i]] = gamma * new[w] + (1.-gamma) * w_avg[i]
            
        ws_avg += [w_avg]   
        
    return new, ws_avg


def AdaMax(w, objective, alpha=.01, beta1=.1, beta2=.001):
    print 'AdaMax', 'alpha:',alpha,'beta1:',beta1,'beta2:',beta2
    g = T.grad(objective.sum(), w, disconnected_inputs='warn')
    
    new = OrderedDict()
    
    for i in range(len(w)):
        #gi = T.switch(T.isnan(gi),T.zeros_like(gi),gi) #remove NaN's
        mom1 = G.sharedf(w[i].get_value() * 0.)
        _max = G.sharedf(w[i].get_value() * 0.)
        new[mom1] = (1-beta1) * mom1 + beta1 * g[i]
        new[_max] = T.maximum((1-beta2)*_max, abs(g[i]) + 1e-8)
        new[w[i]] = w[i] + alpha *  new[mom1] / new[_max]
                
    return new

# AdaMax that averages over multiple minibatches
def AdaMax2(w, objective, alpha=.01, beta1=.1, beta2=.001, n_accum=2):
    print 'AdaMax2', 'alpha:',alpha,'beta1:',beta1,'beta2:',beta2, 'n_accum:', n_accum
    g = T.grad(objective.sum(), w, disconnected_inputs='warn')
    
    new = OrderedDict()
    
    from theano.ifelse import ifelse
    it = G.sharedf(0.)
    new[it] = it + 1
    reset = T.eq(T.mod(new[it],n_accum), 0)
    update = T.eq(T.mod(new[it],n_accum), n_accum-1)

    for i in range(len(w)):
        mom1 = G.sharedf(w[i].get_value() * 0.)
        _max = G.sharedf(w[i].get_value() * 0.)
        g_sum = G.sharedf(w[i].get_value() * 0.)
        
        #gi = T.switch(T.isnan(gi),T.zeros_like(gi),gi) #remove NaN's
        new[g_sum] = ifelse(reset, g[i], g_sum + g[i])
        new[mom1] = ifelse(update, (1-beta1) * mom1 + beta1 * new[g_sum], mom1)
        new[_max] = ifelse(update, T.maximum((1-beta2)*_max, abs(new[g_sum]) + 1e-8), _max)
        new[w[i]] = ifelse(update, w[i] + alpha *  new[mom1] / new[_max], w[i])
                
    return new

# AdaMax that keeps running average of parameter
def AdaMaxAvg(ws, ws_avg, objective, alpha=.01, beta1=.1, beta2=.001, update_keys=None, disconnected_inputs='raise'):
    print 'AdaMax_Avg', 'alpha:',alpha,'beta1:',beta1,'beta2:',beta2
    
    gs = G.ndict.T_grad(objective.sum(), ws, disconnected_inputs=disconnected_inputs) #warn/raise
    
    if update_keys is None:
        update_keys = [ws[j].keys() for j in range(len(ws))]
    
    new = OrderedDict()
    for j in range(len(ws)):
        if ws_avg is not None:
            w_avg = ws_avg[j]
        for i in update_keys[j]:
            _w = ws[j][i]
            _g = gs[j][i]
            #_g = T.switch(T.isnan(_g),T.zeros_like(_g),_g) #remove NaN's
            mom1 = G.sharedf(_w.get_value() * 0.)
            _max = G.sharedf(_w.get_value() * 0. + 1e-8)
            
            new[mom1] = (1-beta1) * mom1 + beta1 * _g
            new[_max] = T.maximum((1-beta2)*_max, abs(_g) + 1e-8)
            new[_w] = _w + alpha *  new[mom1] / new[_max]
            if ws_avg is not None:
                new[w_avg[i]] = beta2 * _w + (1.-beta2) * w_avg[i]
    return new

# Eve that keeps running average of parameter
def Eve(w, w_avg, f, alpha=.01, beta1=.1, beta2=.001, beta3=0.01, disconnected_inputs='raise'):
    print 'Eve', 'alpha:',alpha,'beta1:',beta1,'beta2:',beta2,'beta3:',beta3

    mom = {}
    _max = {}
    delta = {}
    w_prime = {}
    for i in w:
        mom[i] = G.sharedf(w[i].get_value() * 0.)
        _max[i] = G.sharedf(w[i].get_value() * 0. + 1e-8)
        delta[i] = G.sharedf(w[i].get_value() * 0.)
        w_prime[i] = w[i] + (1-beta1)/beta1 * delta[i]
    
    train_cost = f(w_prime).mean()
    g = G.ndict.T_grad(train_cost, w, disconnected_inputs=disconnected_inputs) #warn/raise
    
    new = OrderedDict()
    for i in w:
        new[mom[i]] = (1-beta1) * mom[i] + beta1 * g[i]
        new[_max[i]] = T.maximum((1-beta2)*_max[i], abs(g[i]) + 1e-8)
        new[delta[i]] = alpha * new[mom[i]] / new[_max[i]]
        new[w[i]] = w[i] + new[delta[i]]
    
    for i in w:
        new[w_avg[i]] = beta3 * w[i] + (1.-beta3) * w_avg[i]
    return train_cost, new
    
# AdaMax that keeps running average of parameter
# Accumulates gradient over n_accum minibatches
def AdaMaxAvg2(ws, objective, alpha=.01, beta1=.1, beta2=.001, beta3=0.01, n_accum=1):
    if n_accum == 1:
        return AdaMaxAvg(ws, objective, alpha, beta1, beta2, beta3)
    print 'AdaMax_Avg2', 'alpha:',alpha,'beta1:',beta1,'beta2:',beta2,'beta3:',beta3,'n_accum:',n_accum
    
    gs = G.ndict.T_grad(objective.sum(), ws, disconnected_inputs='raise')

    new = OrderedDict()
    
    from theano.ifelse import ifelse
    it = G.sharedf(0.)
    new[it] = it + 1
    reset = T.eq(T.mod(it,n_accum), 0)
    update = T.eq(T.mod(it,n_accum), n_accum-1)
    
    ws_avg = []
    for j in range(len(ws)):
        w_avg = {}
        for i in ws[j]:
            _w = ws[j][i]
            _g = gs[j][i]
            #_g = T.switch(T.isnan(_g),T.zeros_like(_g),_g) #remove NaN's
            mom1 = G.sharedf(_w.get_value() * 0.)
            _max = G.sharedf(_w.get_value() * 0.)
            w_avg[i] = G.sharedf(_w.get_value())
            g_sum = G.sharedf(_w.get_value() * 0.)
        
            new[g_sum] = ifelse(reset, _g, g_sum + _g)
            new[mom1] = ifelse(update, (1-beta1) * mom1 + beta1 * new[g_sum], mom1)
            new[_max] = ifelse(update, T.maximum((1-beta2)*_max, abs(new[g_sum]) + 1e-8), _max)
            new[_w] = ifelse(update, _w + alpha *  new[mom1] / new[_max], _w)
            new[w_avg[i]] = ifelse(update, beta3 * new[_w] + (1.-beta3) * w_avg[i], w_avg[i])
        ws_avg += [w_avg]   
    return new, ws_avg


