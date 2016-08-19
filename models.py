import graphy as G
import graphy.nodes as N
import graphy.nodes.rand
import graphy.nodes.conv
import graphy.nodes.ar
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from pyexpat import model

floatX = theano.config.floatX # @UndefinedVariable

# CVAE ResNet layer of deterministic and stochastic units
def cvae_layer(name, prior, posterior, n_h1, n_h2, n_z, depth_ar, downsample, nl, kernel, weightsharing, downsample_type, w):
    
    if False:
        # New such that we can recognize variational params later
        name_q = name+'_q_'
        name_p = name+'_p_'
    else:
        name_q = name
        name_p = name
    
    n_conv_up1 = n_h2+2*n_z
    n_conv_up2 = n_h2+n_z
    
    n_conv_down_posterior = 0
    n_conv_down_prior = n_h2+2*n_z
    
    # Prior
    prior_conv1 = None
    
    if prior in ['diag','diag2']:
        n_conv_down_prior = n_h2+2*n_z
    elif prior == 'made':
        prior_conv1 = N.ar.multiconv2d(name_p+'_prior_conv1', n_z, depth_ar*[n_h2], [n_z,n_z], kernel, False, nl=nl, w=w)
        n_conv_down_prior = n_h2+n_h2
    elif prior == 'bernoulli':
        n_conv_down_prior = n_h2+n_z
        prior_conv1 = N.conv.conv2d(name_p+'_prior_conv1', n_z, n_z, kernel, w=w)
    else:
        raise Exception("Unknown prior")
    
    # Posterior
    posterior_conv1 = None
    posterior_conv2 = None
    posterior_conv3 = None
    posterior_conv4 = None
    
    if posterior == 'up_diag':
        pass
    elif posterior == 'up_iaf1':
        posterior_conv1 = N.ar.conv2d(name_q+'_posterior_conv1', n_z, n_z, kernel, w=w)
    elif posterior == 'up_iaf2':
        posterior_conv1 = N.ar.conv2d(name_q+'_posterior_conv1', n_z, 2*n_z, kernel, w=w)
    
    elif posterior == 'up_iaf1_nl':
        n_conv_up1 = n_h2+2*n_z+n_h2
        posterior_conv1 = N.ar.multiconv2d(name_q+'_posterior_conv1', n_z, depth_ar*[n_h2], n_z, kernel, False, nl=nl, w=w)
    elif posterior == 'up_iaf2_nl':
        n_conv_up1 = n_h2+2*n_z+n_h2
        posterior_conv1 = N.ar.multiconv2d(name_q+'_posterior_conv1', n_z, depth_ar*[n_h2], [n_z,n_z], kernel, False, nl=nl, w=w)
    
#    elif posterior == 'down_diag':
#        n_conv_down1 = n_h2+4*n_z
    elif posterior == 'down_diag':
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z
    elif posterior == 'down_bernoulli':
        n_conv_up2 = n_h2
        n_conv_down_posterior = n_z
    elif posterior == 'down_tim':
        pass
    elif posterior == 'down_iaf1':
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z
        posterior_conv1 = N.ar.conv2d(name_q+'_posterior_conv1', n_z, n_z, kernel, w=w)
    elif posterior == 'down_iaf2':
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z
        posterior_conv1 = N.ar.conv2d(name_q+'_posterior_conv1', n_z, 2*n_z, kernel, w=w)
    elif posterior == 'down_iaf1_nl':
        n_conv_up1 = n_h2+2*n_z+n_h2
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z+n_h2
        posterior_conv1 = N.ar.multiconv2d(name_q+'_posterior_conv1', n_z, depth_ar*[n_h2], n_z, kernel, False, nl=nl, w=w)
    elif posterior == 'down_iaf2_nl':
        n_conv_up1 = n_h2+2*n_z+n_h2
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z+n_h2
        posterior_conv1 = N.ar.multiconv2d(name_q+'_posterior_conv1', n_z, depth_ar*[n_h2], [n_z,n_z], kernel, False, nl=nl, w=w)
    elif posterior == 'down_iaf2_nl2':
        n_conv_up1 = n_h2+2*n_z+n_h2
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z+n_h2
        posterior_conv1 = N.ar.multiconv2d(name_q+'_posterior_conv1', n_z, depth_ar*[n_h2], [n_z,n_z], kernel, False, nl=nl, w=w)
        posterior_conv2 = N.ar.multiconv2d(name_q+'_posterior_conv2', n_z, depth_ar*[n_h2], [n_z,n_z], kernel, True, nl=nl, w=w)
    elif posterior == 'down_iaf1_deep':
        n_conv_up1 = n_h2+2*n_z+n_h2
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z+n_h2
        posterior_conv1 = N.ar.resnet(name_q+'_deepiaf', depth_ar, n_z, n_h2, n_z, kernel, False, nl=nl, weightsharing=weightsharing, w=w)
    elif posterior == 'down_iaf2_deep':
        n_conv_up1 = n_h2+2*n_z+n_h2
        n_conv_up2 = n_h2
        n_conv_down_posterior = 2*n_z+n_h2
        posterior_conv1 = N.ar.resnet(name_q+'_deepiaf', depth_ar, n_z, n_h2, [n_z,n_z], kernel, False, nl=nl, weightsharing=weightsharing, w=w)
    
    #elif posterior == 'iaf_deep1':
    #    extra1 = N.ar.resnet(name+'_posterior_2', depth_iaf, n_z, 2*n_h, n_h, n_z, (3,3), False, nl=nl, w=w)
    #elif posterior == 'iaf_deep2':
    #    extra1 = N.ar.resnet(name+'_posterior_2', depth_iaf, n_z, 2*n_h, n_h, [n_z,n_z], (3,3), False, nl=nl, w=w)
    else:
        raise Exception("Unknown posterior "+posterior)
    
    ds = 1
    if downsample:
        ds = 2
        if downsample_type == 'conv':
            up_conv3 = N.conv.conv2d(name_q+'_up_conv3', n_h1, n_h1, kernel, downsample=ds, w=w)
            down_conv3 = N.conv.conv2d(name_q+'_down_conv3', n_h1, n_h1, kernel, upsample=ds, w=w)
    
    up_nl1 = N.nonlinearity(name_q+"_up_nl1", nl)
    up_conv1 = N.conv.conv2d(name_q+'_up_conv1_'+str(ds), n_h1, n_conv_up1, kernel, downsample=ds, w=w)
    up_nl2 = N.nonlinearity(name_q+"_nl_up2", nl)
    up_conv2 = N.conv.conv2d(name_q+'_up_conv2', n_conv_up2, n_h1, kernel, w=w)
    
    down_nl1 = N.nonlinearity(name_p+"_down_nl1", nl)
    down_conv1 = N.conv.conv2d(name_p+'_down_conv1', n_h1, n_conv_down_prior+n_conv_down_posterior, kernel, w=w)
    down_nl2 = N.nonlinearity(name_p+"_down_nl2", nl)
    down_conv2 = N.conv.conv2d(name_p+'_down_conv2_'+str(ds), n_h2+n_z, n_h1, kernel, upsample=ds, w=w)
    
    up_output = [None]
    qz = [None]
    up_context = [None]
    
    def up(input, w):

        h = up_conv1(up_nl1(input, w), w)
        h_det = h[:,:n_h2,:,:]
        qz_mean = h[:,n_h2:n_h2+n_z,:,:]
        qz_logsd = h[:,n_h2+n_z:n_h2+2*n_z,:,:]
        qz[0] = N.rand.gaussian_diag(qz_mean, 2*qz_logsd)
        if posterior == 'up_diag':
            h = T.concatenate([h_det,qz[0].sample],axis=1)
        elif posterior == 'up_iaf1':
            arw_mean = posterior_conv1(qz[0].sample, w)
            arw_mean *= .1
            qz[0].sample = (qz[0].sample - arw_mean)
            h = T.concatenate([h_det,qz[0].sample],axis=1)
        elif posterior == 'up_iaf2':
            arw_mean_logsd = posterior_conv1(qz[0].sample, w)
            arw_mean = arw_mean_logsd[:,::2,:,:]
            arw_logsd = arw_mean_logsd[:,1::2,:,:]
            arw_mean *= .1
            arw_logsd *= .1
            qz[0].sample = (qz[0].sample - arw_mean) / T.exp(arw_logsd)
            qz[0].logps += arw_logsd
            qz[0].logp += arw_logsd.flatten(2).sum(axis=1)
            h = T.concatenate([h_det,qz[0].sample],axis=1)
        elif posterior == 'up_iaf1_nl':
            context = h[:,n_h2+2*n_z:n_h2+2*n_z+n_h2]
            arw_mean = posterior_conv1(qz[0].sample, context, w)
            arw_mean *= .1
            qz[0].sample = (qz[0].sample - arw_mean)
            h = T.concatenate([h_det,qz[0].sample],axis=1)
        elif posterior == 'up_iaf2_nl':
            context = h[:,n_h2+2*n_z:n_h2+2*n_z+n_h2]
            arw_mean, arw_logsd = posterior_conv1(qz[0].sample, context, w)
            arw_mean *= .1
            arw_logsd *= .1
            qz[0].sample = (qz[0].sample - arw_mean) / T.exp(arw_logsd)
            qz[0].logps += arw_logsd
            qz[0].logp += arw_logsd.flatten(2).sum(axis=1)
            h = T.concatenate([h_det,qz[0].sample],axis=1)
        elif posterior == 'down_tim':
            h = T.concatenate([h_det,qz[0].mean],axis=1)
        elif posterior in ['down_iaf1_nl','down_iaf2_nl','down_iaf2_nl2','down_iaf1_deep','down_iaf2_deep']:
            up_context[0] = h[:,n_h2+2*n_z:n_h2+2*n_z+n_h2]
            h = h_det
        elif posterior in ['down_diag','down_iaf1','down_iaf2','down_bernoulli']:
            h = h_det
        else:
            raise Exception()
        if downsample:
            if downsample_type == 'nn':
                input = N.conv.downsample2d_nearest_neighbour(input, 2)
            elif downsample_type == 'conv':
                input = up_conv3(input, w)
        output = input + .1 * up_conv2(up_nl2(h, w), w)
        up_output[0] = output

        return output
    
    def bernoulli_p(h):
        #p = T.clip(.5+.5*h, 1e-7, 1. - 1e-7)
        p = 1e-7 + (1-2e-7)*T.nnet.sigmoid(h)
        return p
    
    def down_q(input, train, w):
        
        #if name == '1':
        #    print input.tag.test_value
        
        # prior
        h = down_nl1(input, w)
        #h = T.printing.Print('h1'+name)(h)
        h = down_conv1(h, w)
        #h = T.printing.Print('h2'+name)(h)
        
        logqs = 0
        
        # posterior
        if posterior in ['up_diag','up_iaf1','up_iaf2','up_iaf1_nl','up_iaf2_nl']:
            z = qz[0].sample
            logqs = qz[0].logps
        elif posterior == 'down_diag':
            rz_mean = h[:,n_conv_down_prior:n_conv_down_prior+n_z,:,:]
            rz_logsd = h[:,n_conv_down_prior+n_z:n_conv_down_prior+2*n_z,:,:]
            _qz = N.rand.gaussian_diag(qz[0].mean + rz_mean, qz[0].logvar + 2*rz_logsd)
            z = _qz.sample
            logqs = _qz.logps                
        elif posterior == 'down_tim':
            assert prior == 'diag'
            pz_mean = h[:,n_h2:n_h2+n_z,:,:]
            pz_logsd = h[:,n_h2+n_z:n_h2+2*n_z,:,:]
            
            qz_prec = 1./T.exp(qz[0].logvar)
            pz_prec = 1./T.exp(2*pz_logsd)
            rz_prec = qz_prec + pz_prec
            rz_mean = (pz_prec/rz_prec) * pz_mean + (qz_prec/rz_prec) * qz[0].mean
            _qz = N.rand.gaussian_diag(rz_mean, -T.log(rz_prec))
            z = _qz.sample
            logqs = _qz.logps
        elif posterior == 'down_iaf1':
            rz_mean = h[:,n_conv_down_prior:n_conv_down_prior+n_z,:,:]
            rz_logsd = h[:,n_conv_down_prior+n_z:n_conv_down_prior+2*n_z,:,:]
            _qz = N.rand.gaussian_diag(qz[0].mean + rz_mean, qz[0].logvar + 2*rz_logsd)
            z = _qz.sample
            logqs = _qz.logps
            # ARW transform
            arw_mean = posterior_conv1(z, w)
            arw_mean *= .1
            z = (z - arw_mean)
        elif posterior == 'down_iaf2':
            rz_mean = h[:,n_conv_down_prior:n_conv_down_prior+n_z,:,:]
            rz_logsd = h[:,n_conv_down_prior+n_z:n_conv_down_prior+2*n_z,:,:]
            _qz = N.rand.gaussian_diag(qz[0].mean + rz_mean, qz[0].logvar + 2*rz_logsd)
            z = _qz.sample
            logqs = _qz.logps
            # ARW transform
            arw_mean_logsd = posterior_conv1(z, w)
            arw_mean = arw_mean_logsd[:,::2,:,:]
            arw_logsd = arw_mean_logsd[:,1::2,:,:]
            arw_mean *= .1
            arw_logsd *= .1
            z = (z - arw_mean) / T.exp(arw_logsd)
            logqs += arw_logsd
        elif posterior in ['down_iaf1_nl','down_iaf1_deep']:
            rz_mean = h[:,n_conv_down_prior:n_conv_down_prior+n_z,:,:]
            rz_logsd = h[:,n_conv_down_prior+n_z:n_conv_down_prior+2*n_z,:,:]
            _qz = N.rand.gaussian_diag(qz[0].mean + rz_mean, qz[0].logvar + 2*rz_logsd)
            z = _qz.sample
            logqs = _qz.logps
            # ARW transform
            down_context = h[:,n_conv_down_prior+2*n_z:n_conv_down_prior+2*n_z+n_h2,:,:]
            context = up_context[0] + down_context
            arw_mean = posterior_conv1(z, context, w)
            arw_mean *= .1
            z = (z - arw_mean)
        elif posterior in ['down_iaf2_nl','down_iaf2_nl2','down_iaf2_deep']:
            rz_mean = h[:,n_conv_down_prior:n_conv_down_prior+n_z,:,:]
            rz_logsd = h[:,n_conv_down_prior+n_z:n_conv_down_prior+2*n_z,:,:]
            _qz = N.rand.gaussian_diag(qz[0].mean + rz_mean, qz[0].logvar + 2*rz_logsd)
            z = _qz.sample
            logqs = _qz.logps
            # ARW transform
            down_context = h[:,n_conv_down_prior+2*n_z:n_conv_down_prior+2*n_z+n_h2,:,:]
            context = up_context[0] + down_context
            arw_mean, arw_logsd = posterior_conv1(z, context, w)
            arw_mean *= .1
            arw_logsd *= .1
            z = (z - arw_mean) / T.exp(arw_logsd)
            logqs += arw_logsd
            if posterior == 'down_iaf2_nl2':
                arw_mean, arw_logsd = posterior_conv2(z, context, w)
                arw_mean *= .1
                arw_logsd *= .1
                z = (z - arw_mean) / T.exp(arw_logsd)
                logqs += arw_logsd
            
        
        # Prior
        if prior == 'diag':
            pz_mean = h[:,n_h2:n_h2+n_z,:,:]
            pz_logsd = h[:,n_h2+n_z:n_h2+2*n_z,:,:]
            logps = N.rand.gaussian_diag(pz_mean, 2*pz_logsd, z).logps
        elif prior == 'diag2':
            logps = N.rand.gaussian_diag(0*z, 0*z, z).logps
            pz_mean = h[:,n_h2:n_h2+n_z,:,:]
            pz_logsd = h[:,n_h2+n_z:n_h2+2*n_z,:,:]
            z = pz_mean + z * T.exp(pz_logsd)
        elif prior == 'made':
            made_context = h[:,n_h2:2*n_h2,:,:]
            made_mean, made_logsd = prior_conv1(z, made_context, w)
            made_mean *= .1
            made_logsd *= .1
            logps = N.rand.gaussian_diag(made_mean, 2*made_logsd, z).logps
        elif prior == 'bernoulli':
            assert posterior == 'down_bernoulli'
            pz_p = bernoulli_p(h[:,n_h2:n_h2+n_z,:,:])
            logps = z01 * T.log(pz_p) + (1.-z01) * T.log(1.-pz_p)
        else:
            raise Exception()
        
        h_det = h[:,:n_h2,:,:]
        h = T.concatenate([h_det, z], axis=1)
        if downsample:
            if downsample_type == 'nn':
                input = N.conv.upsample2d_nearest_neighbour(input)
            elif downsample_type == 'conv':
                input = down_conv3(input, w)
        
        output = input + .1 * down_conv2(down_nl2(h, w), w)
        
        
        return output, logqs - logps
    
    def down_p(input, eps, w):
        # prior
        h = down_conv1(down_nl1(input, w), w)
        h_det = h[:,:n_h2,:,:]
        if prior in ['diag','diag2']:
            mean_prior = h[:,n_h2:n_h2+n_z,:,:]
            logsd_prior = h[:,n_h2+n_z:n_h2+2*n_z,:,:]
            z = mean_prior + eps * T.exp(logsd_prior)
        elif prior == 'made':
            print "TODO: SAMPLES FROM MADE PRIOR"
            z = eps
        elif prior == 'bernoulli':
            assert posterior == 'down_bernoulli'
            pz_p = bernoulli_p(h[:,n_h2:n_h2+n_z,:,:])
            if False:
                z = N.rand.bernoulli(pz_p).sample
            else:
                print "Alert: Sampling using Gaussian approximation"
                z = pz_p + T.sqrt(pz_p * (1-pz_p)) * eps
            z = prior_conv1(2*z-1, w)
        
        h = T.concatenate([h_det, z], axis=1)
        if downsample:
            if downsample_type == 'nn':
                input = N.conv.upsample2d_nearest_neighbour(input)
            elif downsample_type == 'conv':
                input = down_conv3(input, w)
            
        output = input + .1 * down_conv2(down_nl2(h, w), w)
        return output
        
    def postup(updates, w):
        modules = [up_conv1,up_conv2,down_conv1,down_conv2]
        if downsample and downsample_type == 'conv':
            modules += [up_conv3,down_conv3]
        if prior_conv1 != None:
            modules.append(prior_conv1)
        if posterior_conv1 != None:
            modules.append(posterior_conv1)
        if posterior_conv2 != None:
            modules.append(posterior_conv2)
        if posterior_conv3 != None:
            modules.append(posterior_conv3)
        if posterior_conv3 != None:
            modules.append(posterior_conv4)
        for m in modules:
            updates = m.postup(updates, w)
        return updates
    
    return G.Struct(up=up, down_q=down_q, down_p=down_p, postup=postup, w=w)

# Conv VAE
# - Hybrid deterministic/stochastic ResNet block per layer

def cvae1(shape_x, depths, depth_ar, n_h1, n_h2, n_z, prior='diag', posterior='down_diag', px='logistic', nl='softplus', kernel_x=(5,5), kernel_h=(3,3), kl_min=0, optim='adamax', alpha=0.002, beta1=0.1, beta2=0.001, weightsharing=None, pad_x = 0, data_init=None, downsample_type='nn'):
    _locals = locals()
    _locals.pop('data_init')
    print 'CVAE1 with ', _locals
    #assert posterior in ['diag1','diag2','iaf_linear','iaf_nonlinear']
    assert px in ['logistic','bernoulli']
    w = {} # model params
    if pad_x > 0:
        shape_x[1] += 2*pad_x
        shape_x[2] += 2*pad_x
    
    # Input whitening
    if px == 'logistic':
        w['logsd_x'] = G.sharedf(0.)
    
    # encoder
    x_enc = N.conv.conv2d('x_enc', shape_x[0], n_h1, kernel_x, downsample=2, w=w)
    x_dec = N.conv.conv2d('x_dec', n_h1, shape_x[0], kernel_x, upsample=2, w=w)
    x_dec_nl = N.nonlinearity('x_dec_nl', nl, n_h1, w)
    
    layers = []
    for i in range(len(depths)):
        layers.append([])
        for j in range(depths[i]):
            downsample = (i > 0 and j == 0)
            if weightsharing is None or not weightsharing:
                name = str(i)+'_'+str(j)
            elif weightsharing == 'all':
                name = '[sharedw]'+str(i)+'_'+str(j)+'[/sharedw]'
            elif weightsharing == 'acrosslevels':
                name = '[sharedw]'+str(i)+'[/sharedw]'+'_'+str(j)
            elif weightsharing == 'withinlevel':
                name = '[sharedw]'+str(i)+'[/sharedw]'+'_'+str(j)
            else:
                raise Exception()
            layers[i].append(cvae_layer(name, prior, posterior, n_h1, n_h2, n_z, depth_ar, downsample, nl, kernel_h, False, downsample_type, w))
    
    # top-level value
    w['h_top'] = G.sharedf(np.zeros((n_h1,)))
    
    # Initialize variables
    x = T.tensor4('x', dtype='uint8')
    x.tag.test_value = data_init['x']
    n_batch_test = data_init['x'].shape[0]
    _x = T.clip((x + .5) / 256., 0, 1)
    #_x = T.clip(x / 255., 0, 1)
    
    if pad_x > 0:
        _x = N.conv.pad2d(_x, pad_x)
    
    # Objective function
    def f_encode_decode(w, train=True):
        
        results = {}
        
        h = x_enc(_x - .5, w)
        
        obj_kl = G.sharedf(0.)
        
        # bottom-up encoders
        for i in range(len(depths)):
            for j in range(depths[i]):
                h = layers[i][j].up(h, w)
        
        # top-level activations
        h = T.tile(w['h_top'].dimshuffle('x',0,'x','x'), (_x.shape[0],1,shape_x[1]/2**len(depths), shape_x[2]/2**len(depths)))
        
        # top-down priors, posteriors and decoders
        for i in list(reversed(range(len(depths)))):
            for j in list(reversed(range(depths[i]))):
                h, kl = layers[i][j].down_q(h, train, w)
                kl_sum = kl.sum(axis=(1,2,3))
                results['cost_z'+str(i).zfill(3)+'_'+str(j).zfill(3)] = kl_sum
                # Constraint: Minimum number of bits per featuremap, averaged across minibatch
                if kl_min > 0:
                    if True:
                        kl = kl.sum(axis=(2,3)).mean(axis=0,dtype=G.floatX)
                        obj_kl += T.maximum(np.asarray(kl_min,G.floatX), kl).sum(dtype=G.floatX)
                    else:
                        kl = T.maximum(np.asarray(kl_min,G.floatX), kl.sum(axis=(2,3))).sum(axis=1,dtype=G.floatX)
                        obj_kl += kl
                else:
                    obj_kl += kl_sum
        
        output = .1 * x_dec(x_dec_nl(h, w), w)
        
        # empirical distribution
        if px == 'logistic':
            mean_x = T.clip(output+.5, 0+1/512., 1-1/512.)
            logsd_x = 0*mean_x + w['logsd_x']
            obj_logpx = N.rand.discretized_logistic(mean_x, logsd_x, 1/256., _x).logp
            #obj_z = T.printing.Print('obj_z')(obj_z)
            obj = obj_logpx - obj_kl
            # Compute the bits per pixel
            obj *= (1./np.prod(shape_x) * 1./np.log(2.)).astype('float32')
            
            #if not '__init' in w:
            #    raise Exception()
        
        elif px == 'bernoulli':
            prob_x = T.nnet.sigmoid(output)
            prob_x = T.maximum(T.minimum(prob_x, 1-1e-7), 1e-7)
            #prob_x = T.printing.Print('prob_x')(prob_x)
            obj_logpx = N.rand.bernoulli(prob_x, _x).logp
            
            #obj_logqz = T.printing.Print('obj_logqz')(obj_logqz)
            #obj_logpz = T.printing.Print('obj_logpz')(obj_logpz)
            #obj_logpx = T.printing.Print('obj_logpx')(obj_logpx)
            obj = obj_logpx - obj_kl
            #obj = T.printing.Print('obj')(obj)
        
        results['cost_x'] = -obj_logpx
        results['cost'] = -obj
        return results

    # Turns Gaussian noise 'eps' into a sample 
    def f_decoder(eps, w):
        
        # top-level activations
        h = T.tile(w['h_top'].dimshuffle('x',0,'x','x'), (eps['eps_0_0'].shape[0],1,shape_x[1]/2**len(depths), shape_x[2]/2**len(depths)))
        
        # top-down priors, posteriors and decoders
        for i in list(reversed(range(len(depths)))):
            for j in list(reversed(range(depths[i]))):
                h = layers[i][j].down_p(h, eps['eps_'+str(i)+'_'+str(j)], w)
        
        output = .1 * x_dec(x_dec_nl(h, w), w)
        
        if px == 'logistic':
            mean_x = T.clip(output+.5, 0+1/512., 1-1/512.)
        elif px == 'bernoulli':
            mean_x = T.nnet.sigmoid(output)
        
        image = (256.*mean_x).astype('uint8')
        if pad_x > 0:
            image = image[:,:,pad_x:-pad_x,pad_x:-pad_x]
        
        return image
    
    def f_eps(n_batch, w):
        eps = {}
        for i in range(len(depths)):
            for j in range(depths[i]):
                eps['eps_'+str(i)+'_'+str(j)] = G.rng_curand.normal((n_batch,n_z,shape_x[1]/2**(i+1),shape_x[2]/2**(i+1)),dtype=floatX)
        return eps
            
    def postup(updates, w):
        nodes = [x_enc,x_dec]
        for n in nodes:
            updates = n.postup(updates, w)
        for i in range(len(depths)):
            for j in range(depths[i]):
                updates = layers[i][j].postup(updates, w)
        
        return updates
    
    # Compile init function
    if data_init != None:
        w['__init'] = OrderedDict()
        f_encode_decode(w)
        w.pop('__init')
        #for i in w: print i, abs(w[i].get_value()).min(), abs(w[i].get_value()).max(), abs(w[i].get_value()).mean()
    
    # Compile training function
        
    #todo: replace postup with below
    #w['_updates'] = updates
    #f_cost(w)
    #updates = w.pop('_updates')
    
    
    w_avg = {i: G.sharedf(w[i].get_value()) for i in w}
    
    def lazy(f):
        def newf(*args, **kws):
            if not hasattr(f, 'cache'):
                f.cache = f()
            return f.cache(*args, **kws)
        return newf
    
    @lazy
    def f_train():
        if optim == 'adamax':
            train_cost = f_encode_decode(w)['cost']
            updates = G.misc.optim.AdaMaxAvg([w],[w_avg], train_cost, alpha=-alpha, beta1=beta1, beta2=beta2, disconnected_inputs='ignore')
        elif optim == 'eve':
            f = lambda w: f_encode_decode(w)['cost']
            train_cost, updates = G.misc.optim.Eve(w, w_avg, f, alpha=-alpha, beta1=beta1, beta2=beta2, disconnected_inputs='ignore')
        updates = postup(updates, w)
        return G.function({'x':x}, train_cost, updates=updates, lazy=lazy)    

    @lazy
    def f_train_q():
        keys_q = []
        for i in w:
            if '_q_' in i: keys_q.append(i)
        train_cost = f_encode_decode(w)['cost']
        updates = G.misc.optim.AdaMaxAvg([w],None, train_cost, alpha=-alpha, beta1=beta1, beta2=beta2, update_keys=keys_q, disconnected_inputs='ignore')
        updates = postup(updates, w)
        return G.function({'x':x}, train_cost, updates=updates, lazy=lazy)    
    
    # Compile evaluation function
    @lazy
    def f_eval():
        results = f_encode_decode(w_avg, False)
        return G.function({'x':x}, results)
    
    # Compile epsilon generating function
    @lazy
    def f_eps_():
        n_batch = T.lscalar()
        n_batch.tag.test_value = 16
        eps = f_eps(n_batch, w)
        return G.function({'n_batch':n_batch}, eps, lazy=lazy)
    
    # Compile sampling function
    @lazy
    def f_decode():
        eps = {}
        for i in range(len(depths)):
            for j in range(depths[i]):
                eps['eps_'+str(i)+'_'+str(j)] = T.tensor4('eps'+str(i))
                eps['eps_'+str(i)+'_'+str(j)].tag.test_value = np.random.randn(n_batch_test,n_z,shape_x[1]/2**(i+1),shape_x[2]/2**(i+1)).astype(floatX)
        image = f_decoder(eps, w_avg)
        return G.function(eps, image, lazy=lazy)
    
    return G.Struct(train=f_train, eval=f_eval, decode=f_decode, eps=f_eps_, w=w, w_avg=w_avg)

# Fully-connected VAE
# - Hybrid deterministic/stochastic ResNet block per layer

def fcvae(shape_x, depth_model, depth_ar, n_h1, n_h2, n_z, posterior, px='logistic', nl='softplus', alpha=0.002, beta1=0.1, beta2=0.001, share_w=False, data_init=None):
    _locals = locals()
    _locals.pop('data_init')
    print 'CVAE9 with ', _locals
    #assert posterior in ['diag1','diag2','iaf_linear','iaf_nonlinear']
    assert px in ['logistic','bernoulli']
    w = {} # model params
    
    kernel_h = (1,1)
    n_x = shape_x[0]*shape_x[1]*shape_x[2]
    
    # Input whitening
    if px == 'logistic':
        w['logsd_x'] = G.sharedf(0.)
    
    # encoder
    x_enc = N.conv.conv2d('x_enc', n_x, n_h1, (1,1), w=w)
    x_dec = N.conv.conv2d('x_dec', n_h1, n_x, (1,1), w=w)
    x_dec_nl = N.nonlinearity('x_dec_nl', nl, n_h1, w)
    
    layers = []
    for i in range(depth_model):
        name = str(i)
        if share_w:
            name = '[sharedw]'+str(i)+'[/sharedw]'
        layers.append(cvae_layer(name, posterior, n_h1, n_h2, n_z, depth_ar, False, nl, kernel_h, share_w, w))
    
    # top-level value
    #w['h_top'] = G.sharedf(np.zeros((n_h1,)))
    w['h_top'] = G.sharedf(np.random.normal(0,0.01,size=(n_h1,)))
    
    # Initialize variables
    x = T.tensor4('x')
    x.tag.test_value = data_init['x']
    n_batch_test = data_init['x'].shape[0]
    _x = T.clip(x / 255., 0, 1)
    
    # Objective function
    def f_cost(w, train=True):
        
        results = {}
        
        h = x_enc(_x.reshape((-1,n_x,1,1)) - .5, w)
        
        obj_logpz = 0
        obj_logqz = 0
        
        # bottom-up encoders
        for i in range(depth_model):
            h = layers[i].up(h, w)
        
        # top-level activations
        h = T.tile(w['h_top'].dimshuffle('x',0,'x','x'), (_x.shape[0],1,1,1))
        
        # top-down priors, posteriors and decoders
        for i in list(reversed(range(depth_model))):
            h, _obj_logqz, _obj_logpz = layers[i].down_q(h, train, w)
            obj_logqz += _obj_logqz
            obj_logpz += _obj_logpz
            results['cost_z'+str(i).zfill(3)] = _obj_logqz - _obj_logpz
        
        output = .1 * x_dec(x_dec_nl(h, w), w).reshape((-1,shape_x[0],shape_x[1],shape_x[2]))
        
        # empirical distribution
        if px == 'logistic':
            mean_x = T.clip(output, -.5, .5)
            logsd_x = 0*mean_x + w['logsd_x']
            obj_logpx = N.rand.discretized_logistic(mean_x, logsd_x, 1/255., _x - .5).logp
            
            obj = obj_logpz - obj_logqz + obj_logpx
            # Compute the bits per pixel
            obj *= (1./np.prod(shape_x) * 1./np.log(2.)).astype('float32')
            
        elif px == 'bernoulli':
            prob_x = T.nnet.sigmoid(output)
            prob_x = T.minimum(prob_x, 1-1e-7)
            prob_x = T.maximum(prob_x, 1e-7)
            #prob_x = T.printing.Print('prob_x')(prob_x)
            obj_logpx = N.rand.bernoulli(prob_x, _x).logp
            
            #obj_logqz = T.printing.Print('obj_logqz')(obj_logqz)
            #obj_logpz = T.printing.Print('obj_logpz')(obj_logpz)
            #obj_logpx = T.printing.Print('obj_logpx')(obj_logpx)
            obj = obj_logpz - obj_logqz + obj_logpx
            #obj = T.printing.Print('obj')(obj)
        
        results['cost_x'] = -obj_logpx
        results['cost'] = -obj
        return results
        
        #print 'obj_logpz', obj_logpz.tag.test_value
        #print 'obj_logqz', obj_logqz.tag.test_value
        #print 'obj_logpx', obj_x.tag.test_value
        #obj_logpz = T.printing.Print('obj_logpz')(obj_logpz)
        #obj_logqz = T.printing.Print('obj_logqz')(obj_logqz)
        #obj_x = T.printing.Print('obj_logpx')(obj_x)

        
        
    
    # Turns Gaussian noise 'eps' into a sample 
    def f_decoder(eps, w):

        # top-level activations
        h = T.tile(w['h_top'].dimshuffle('x',0,'x','x'), (eps['eps_0'].shape[0],1,1,1))
        
        # top-down priors, posteriors and decoders
        for i in list(reversed(range(depth_model))):
            h = layers[i].down_p(h, eps['eps_'+str(i)], w)
        
        output = .1 * x_dec(x_dec_nl(h, w), w).reshape((-1,shape_x[0],shape_x[1],shape_x[2]))
        if px == 'logistic':
            mean_x = T.clip(output[:,:,:,:] + .5, 0, 1)
        elif px == 'bernoulli':
            mean_x = T.nnet.sigmoid(output)
        image = (255.*T.clip(mean_x, 0, 1)).astype('uint8')
        return image
    
    def f_eps(n_batch, w):
        eps = {}
        for i in range(depth_model):
            eps['eps_'+str(i)] = G.rng_curand.normal((n_batch,n_z,1,1),dtype=floatX)
        return eps
            
    def postup(updates, w):
        nodes = [x_enc,x_dec]
        for n in nodes:
            updates = n.postup(updates, w)
        for i in range(depth_model):
            updates = layers[i].postup(updates, w)
        
        return updates
    
    # Compile init function
    if data_init != None:
        w['__init'] = OrderedDict()
        f_cost(w)
        w.pop('__init')
        #for i in w: print i, abs(w[i].get_value()).min(), abs(w[i].get_value()).max(), abs(w[i].get_value()).mean()
    
    # Compile training function
    results = f_cost(w)
    updates, (w_avg,) = G.misc.optim.AdaMaxAvg([w], results['cost'], alpha=-alpha, beta1=beta1, beta2=beta2, disconnected_inputs='ignore')
    #todo: replace postup with below
    #w['_updates'] = updates
    #f_cost(w)
    #updates = w.pop('_updates')
    
    updates = postup(updates, w)
    f_train = G.function({'x':x}, results['cost'], updates=updates)
    
    # Compile evaluation function
    results = f_cost(w_avg, False)
    f_eval = G.function({'x':x}, results)
    
    # Compile epsilon generating function
    n_batch = T.lscalar()
    n_batch.tag.test_value = 16
    eps = f_eps(n_batch, w)
    f_eps = G.function({'n_batch':n_batch}, eps)
    
    # Compile sampling function
    eps = {}
    for i in range(depth_model):
        eps['eps_'+str(i)] = T.tensor4('eps'+str(i))
        eps['eps_'+str(i)].tag.test_value = np.random.randn(n_batch_test,n_z,1,1).astype(floatX)
    image = f_decoder(eps, w_avg)
    f_decode = G.function(eps, image)
    
    return G.Struct(train=f_train, eval=f_eval, decode=f_decode, eps=f_eps, w=w, w_avg=w_avg)

