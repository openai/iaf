import theano.tensor as T
import numpy as np
import graphy as G
import graphy.nodes as N
import graphy.nodes.conv

# hyperparams
logscale = True #Really works better!
logscale_scale = 3.
init_stdev = .1
maxweight = 0.
bn = False #mean-only batchnorm
do_constant_rescale = False

# auto-regressive linear layer
def linear(name, n_in, n_out, diagonalzeros, l2norm=True, w={}):
    assert n_in % n_out == 0 or n_out % n_in == 0
        
    mask = np.ones((n_in, n_out),dtype=G.floatX)
    if n_out >= n_in:
        k = n_out / n_in
        for i in range(n_in):
            mask[i+1:,i*k:(i+1)*k] = 0
            if diagonalzeros:
                mask[i:i+1,i*k:(i+1)*k] = 0
    else:
        k = n_in / n_out
        for i in range(n_out):
            mask[(i+1)*k:,i:i+1] = 0
            if diagonalzeros:
                mask[i*k:(i+1)*k:,i:i+1] = 0
    
    # L2 normalization of weights
    def l2normalize(_w, axis=0):
        if diagonalzeros:
            # to prevent NaN gradients
            # TODO: smarter solution (also see below)
            if n_out >= n_in:
                _w = T.set_subtensor(_w[:,:n_out/n_in], 0.)
            else:
                _w = T.set_subtensor(_w[:,:1], 0.)
        targetnorm = 1.
        norm = T.sqrt((_w**2).sum(axis=axis, keepdims=True))
        norm += 1e-8 
        new_w = _w * (targetnorm / norm)
        return new_w
    def maxconstraint(_w):
        return _w * (maxweight / T.maximum(maxweight, abs(_w).max(axis=0, keepdims=True)))
    
    w[name+'_w'] = G.sharedf(mask * 0.05 * np.random.randn(n_in, n_out))
    if maxweight > 0:
        w[name+'_w'].set_value(maxconstraint(w[name+'_w']).tag.test_value)
    
    w[name+'_b'] = G.sharedf(np.zeros((n_out,)))
    if l2norm:
        if logscale:
            w[name+'_s'] = G.sharedf(np.zeros((n_out,)))
        else:
            w[name+'_s'] = G.sharedf(np.ones((n_out,)))
    elif do_constant_rescale:
        print 'WARNING: constant rescale, these weights arent saved'
        constant_rescale = G.sharedf(np.zeros((n_out,)))
    
    
    def f(h, w):
        _input = h
        _w = mask * w[name+'_w']
        if l2norm:
            _w = l2normalize(_w)
        h = T.dot(h, _w)
        if l2norm:
            if logscale:
                h *= T.exp(logscale_scale*w[name+'_s'])
            else:
                h *= abs(w[name+'_s'])
        elif do_constant_rescale:
            h *= T.exp(constant_rescale)
        
        h += w[name+'_b']
        
        if '__init' in w:
            # Std
            std = (1./init_stdev) * h.std(axis=0)
            std += (std <= 0)
            std += 1e-8
            if name+'_s' in w:
                if logscale:
                    w[name+'_s'].set_value(-T.log(std).tag.test_value/logscale_scale)
                else:
                    w[name+'_s'].set_value((1./std).tag.test_value)
            elif do_constant_rescale:
                constant_rescale.set_value(-T.log(std).tag.test_value)
                #w[name+'_w'].set_value((_w / std.dimshuffle('x',0)).tag.test_value)
                
            h /= std.dimshuffle('x',0)
            
            # Mean
            mean = h.mean(axis=0)
            w[name+'_b'].set_value(-mean.tag.test_value)
            h -= mean.dimshuffle('x',0)
        
            #print name, w[name+'_w'].get_value().mean(), w[name+'_w'].get_value().std(), w[name+'_w'].get_value().max()
        
        #print name, abs(h).max().tag.test_value, abs(h).min().tag.test_value
        #h = T.printing.Print(name)(h)
        
        return h
    
    # Post updates: normalize weights to unit L2 norm
    def postup(updates, w):
        updates[w[name+'_w']] = mask * updates[w[name+'_w']]
        if l2norm and maxweight>0.:
            updates[w[name+'_w']] = maxconstraint(updates[w[name+'_w']])
        return updates
    
    return G.Struct(__call__=f, postup=postup, w=w)

# Auto-Regressive MLP with l2 normalization
# n_in is an int
# n_h is a list of ints
# n_out is an int or list of ints
# nl_h: nonlinearity of hidden units
def mlp(name, n_in, n_context, n_h, n_out, nl, w={}):
    
    if not isinstance(n_out, list) and isinstance(n_out, int):
        n_out = [n_out]
    
    if n_context > 0:
        # parameters for context input
        linear_context = N.linear_l2(name+'_context', n_context, n_h[0], w)
        
    # parameters for hidden units
    nh = [n_in]+n_h
    linear_h = []
    f_nl_h = []
    for i in range(len(n_h)):
        s = name+'_'+str(i)
        linear_h.append(linear(s, nh[i], nh[i+1], False, True, w))
        f_nl_h.append(N.nonlinearity(s+'_nl', nl, (nh[i+1],), w))
        
    # parameters for output
    linear_out = []
    for i in range(len(n_out)):
        s = name+'_out_'+str(i)
        linear_out.append(linear(s, n_h[-1], n_out[i], True, True, w))
    
    def f(h, h_context, w, return_hiddens=False):
        # h_context can be None if n_context == 0
        
        hiddens = []
        for i in range(len(n_h)):
            h = linear_h[i](h, w)
            if i == 0 and n_context > 0:
                h += linear_context(h_context, w)
            h = f_nl_h[i](h, w)
            hiddens.append(h)
        
        out = []
        for i in range(len(n_out)):
            _out = linear_out[i](h, w)
            out.append(_out)

        if len(n_out) == 1: out = out[0]

        if return_hiddens:
            return hiddens, out
        
        return out
    
    def postup(updates, w):
        if n_context > 0:
            updates = linear_context.postup(updates, w)
        for l in linear_h: updates = l.postup(updates, w)
        for l in linear_out: updates = l.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)


def msconv2d(name, n_scales, n_in, n_out, size_kernel=(3,3), zerodiagonal=True, flipmask=False, pad_channel=True, border_mode='valid', w={}):
    convs = [conv2d(name+"_s"+str(i), n_in, n_out, size_kernel, zerodiagonal, flipmask, pad_channel, border_mode, w) for i in range(n_scales)]
    def f(h, w):
        results = []
        for i in range(n_scales-1):
            results.append(convs[i](h, w))
            h = N.conv.downsample2d_nearest_neighbour(h, scale=2)
        result = convs[-1](h, w)
        for i in range(n_scales-1):
            result = N.conv.upsample2d_nearest_neighbour(result)
            result += results[-1-i]
        return result
    
    def postup(updates, w):
        for conv in convs:
            updates = conv.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

def conv2d(name, n_in, n_out, size_kernel=(3,3), zerodiagonal=True, flipmask=False, pad_channel=True, border_mode='valid', zeroinit=False, l2norm=True, w={}):
    
    do_scale = False
    if zeroinit:
        l2norm = False
        do_scale = True
    
    if not pad_channel:
        border_mode = 'same'
        print 'No pad_channel, changing border_mode to same'
        
    #if 'whitener' not in name:
    #    pad_channel = False
    #    border_mode = 'same'
    
    if '[sharedw]' in name and '[/sharedw]' in name:
        name_w = name
        pre, b = name.split("[sharedw]")
        c, post = b.split("[/sharedw]")
        name_w = pre+"[s]"+post
        name = pre+c+post # Don't share the bias and scales
        #name = name_w # Also share the bias and scales
    else:
        name_w = name
    
    assert border_mode in ['valid','full','same']
    
    _n_in = n_in
    
    if pad_channel:
        if size_kernel[0] > 1 or size_kernel[1] > 1:
            assert size_kernel[0] == size_kernel[1]
            assert border_mode == 'valid'
            _n_in += 1
        else:
            pad_channel = False
    
    if border_mode == 'same':
        assert size_kernel[0]%2 == 1
        border_mode = ((size_kernel[0]-1)/2,(size_kernel[1]-1)/2)
    
    if True:
        # Build autoregressive mask
        l = (size_kernel[0]-1)/2
        m = (size_kernel[1]-1)/2
        mask = np.ones((n_out, _n_in, size_kernel[0], size_kernel[1]),dtype=G.floatX)
        mask[:,:,:l,:] = 0
        mask[:,:,l,:m] = 0
        
        if n_out >= n_in:
            assert n_out%n_in == 0
            k = n_out / n_in
            for i in range(n_in):
                mask[i*k:(i+1)*k,i+1:,l,m] = 0
                if zerodiagonal:
                    mask[i*k:(i+1)*k,i:i+1,l,m] = 0
        else:
            assert n_in%n_out == 0
            k = n_in / n_out
            for i in range(n_out):
                mask[i:i+1,(i+1)*k:,l,m] = 0
                if zerodiagonal:
                    mask[i:i+1,i*k:(i+1)*k:,l,m] = 0
        if flipmask:
            mask = mask[::-1,::-1,::-1,::-1]
    
    
    def l2normalize(kerns):
        if zerodiagonal:
            # to prevent NaN gradients
            # TODO: smarter solution (also see below)
            l = (size_kernel[0]-1)/2
            m = (size_kernel[1]-1)/2
            if n_out >= n_in:
                kerns = T.set_subtensor(kerns[:n_out/n_in,:,l,m], 0.)
            else:
                kerns = T.set_subtensor(kerns[:1,:,l,m], 0.)
        
        targetnorm = 1.
        norm = T.sqrt((kerns**2).sum(axis=(1,2,3), keepdims=True))
        norm += 1e-8
        return kerns * (targetnorm / norm)
    def maxconstraint(kerns):
        return kerns * (maxweight / T.maximum(maxweight, abs(kerns).max(axis=(1,2,3), keepdims=True)))

    if zeroinit:
        w[name_w+'_w'] = G.sharedf(np.zeros((n_out, _n_in, size_kernel[0], size_kernel[1])))
    else:
        w[name_w+'_w'] = G.sharedf(mask * 0.05*np.random.randn(n_out, _n_in, size_kernel[0], size_kernel[1]))
        if maxweight > 0:
            w[name_w+'_w'].set_value(maxconstraint(w[name_w+'_w']).tag.test_value)
    
    w[name+'_b'] = G.sharedf(np.zeros((n_out,)))

    if l2norm or do_scale:
        if logscale:
            w[name+'_s'] = G.sharedf(np.zeros((n_out,)))
        else:
            w[name+'_s'] = G.sharedf(np.ones((n_out,)))
    elif do_constant_rescale:
        print 'WARNING: constant rescale, these weights arent saved'
        constant_rescale = G.sharedf(np.ones((n_out,)))
    
    
    def f(h, w):
        input_shape = h.tag.test_value.shape[1:]
        
        _input = h
        
        if pad_channel:
            h = N.conv.pad2dwithchannel(h, size_kernel)
        
        kerns = mask * w[name_w+'_w']
        if l2norm:
            kerns = l2normalize(kerns)
        if l2norm or do_scale:
            if logscale:
                kerns *= T.exp(logscale_scale*w[name+'_s']).dimshuffle(0,'x','x','x')
            else:
                kerns *= w[name+'_s'].dimshuffle(0,'x','x','x')
        elif do_constant_rescale:
            kerns *= constant_rescale.dimshuffle(0,'x','x','x')
        
        h = N.conv.dnn_conv(h, kerns, border_mode=border_mode)
        
        # Center
        if bn: # mean-only batch norm
            h -= h.mean(axis=(0,2,3), keepdims=True)
        
        h += w[name+'_b'].dimshuffle('x',0,'x','x')
        
        if '__init' in w and not zeroinit:
            
            # Std
            data_std = h.std(axis=(0,2,3))
            num_zeros = (data_std.tag.test_value == 0).sum()
            if num_zeros > 0:
                print "Warning: Stdev=0 for "+str(num_zeros)+" features in "+name+". Skipping data-dependent init."
            else:
                if name+'_s' in w:
                    if logscale:
                        w[name+'_s'].set_value(-T.log(data_std).tag.test_value/logscale_scale)
                    else:
                        w[name+'_s'].set_value((1./data_std).tag.test_value)
                elif do_constant_rescale:
                    constant_rescale.set_value((1./data_std).tag.test_value)
                    #w[name+'_w'].set_value((kerns / std.dimshuffle(0,'x','x','x')).tag.test_value)
                
                h /= data_std.dimshuffle('x',0,'x','x')
                
                # Mean
                mean = h.mean(axis=(0,2,3))
                w[name+'_b'].set_value(-mean.tag.test_value)
                h -= mean.dimshuffle('x',0,'x','x')
                
            #print name, w[name+'_w'].get_value().mean(), w[name+'_w'].get_value().std(), w[name+'_w'].get_value().max()
        
        if not '__init' in w:
            output_shape = h.tag.test_value.shape[1:]
            print 'ar.conv2d', name, input_shape, output_shape, size_kernel, zerodiagonal, flipmask, pad_channel, border_mode, zeroinit, l2norm
        
        #print name, abs(h).max().tag.test_value, abs(h).min().tag.test_value
        #h = T.printing.Print(name)(h)
        
        return h
    
    # Normalize weights to _norm L2 norm
    # TODO: check whether only_upper_bounds here really helps
    # (the effect is a higher learning rate in the beginning of training)
    def postup(updates, w):
        updates[w[name_w+'_w']] = mask * updates[w[name_w+'_w']]
        if l2norm and maxweight>0.:
            updates[w[name_w+'_w']] = maxconstraint(updates[w[name_w+'_w']])
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# Auto-Regressive convnet with l2 normalization
def multiconv2d(name, n_in, n_h, n_out, size_kernel, flipmask, nl='relu', w={}):
    
    if not isinstance(n_out, list) and isinstance(n_out, int):
        n_out = [n_out]
    
    # parameters for hidden units
    sizes = [n_in]+n_h
    conv_h = []
    f_nl_h = []
    for i in range(len(n_h)):
        conv_h.append(conv2d(name+'_'+str(i), sizes[i], sizes[i+1], size_kernel, False, flipmask, w=w))
        f_nl_h.append(N.nonlinearity(name+'_'+str(i)+'_nl', nl, sizes[i+1], w=w))
    
    # parameters for output
    conv_out = []
    for i in range(len(n_out)):
        conv_out.append(conv2d(name+'_out_'+str(i), sizes[-1], n_out[i], size_kernel, True, flipmask, w=w))
    
    def f(h, context, w, return_hiddens=False):
        # h_context can be None if n_context == 0
        
        hiddens = []
        for i in range(len(n_h)):
            h = conv_h[i](h, w) # + context
            if i == 0: h += context
            h = f_nl_h[i](h, w)
            hiddens.append(h)
        
        out = []
        for i in range(len(n_out)):
            _out = conv_out[i](h, w)
            out.append(_out)

        if len(n_out) == 1: out = out[0]
        
        if return_hiddens:
            return hiddens, out
        
        return out
    
    def postup(updates, w):
        for l in conv_h: updates = l.postup(updates, w)
        for l in conv_out: updates = l.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)



# ResNet V3 layer
def resnet_layer_a(name, n_feats, nl='elu', w={}):

    f_nl1 = N.nonlinearity(name+"_nl1", nl)
    f_nl2 = N.nonlinearity(name+"_nl2", nl)
    
    # either no change in shape, or subsampling
    conv1 = conv2d(name+'_conv1', n_feats, n_feats, (3,3), zerodiagonal=False, w=w)
    conv2 = conv2d(name+'_conv2', n_feats, n_feats, (3,3), zerodiagonal=False, w=w)
    
    def f(_input, w):
        h = f_nl1(_input)
        h = f_nl2(conv1(h, w))
        h = conv2(h, w)
        return _input + .1 * h
    
    def postup(updates, w):
        updates = conv1.postup(updates, w)
        updates = conv2.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# ResNet V3 layer
def resnet_layer_b(name, n_feats, factor=4, nl='elu', w={}):

    f_nl1 = N.nonlinearity(name+"_nl1", nl)
    f_nl2 = N.nonlinearity(name+"_nl2", nl)
    f_nl3 = N.nonlinearity(name+"_nl3", nl)
    
    # either no change in shape, or subsampling
    conv1 = conv2d(name+'_conv1', n_feats, n_feats/factor, (1,1), zerodiagonal=False, w=w)
    conv2 = conv2d(name+'_conv2', n_feats/factor, n_feats/factor, (3,3), zerodiagonal=False, w=w)
    conv3 = conv2d(name+'_conv3', n_feats/factor, n_feats, (1,1), zerodiagonal=False, w=w)
    
    def f(_input, w):
        h = f_nl1(_input)
        h = f_nl2(conv1(h, w))
        h = f_nl3(conv2(h, w))
        h = conv3(h, w)
        return _input + .1 * h
        
    def postup(updates, w):
        updates = conv1.postup(updates, w)
        updates = conv2.postup(updates, w)
        updates = conv3.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# Auto-Regressive convnet with l2 normalization
def resnet(name, depth, n_in, n_h, n_out, size_kernel=(3,3), flipmask=False, nl='elu', layertype='a', factor=4, weightsharing=False, w={}):
    
    if not isinstance(n_out, list) and isinstance(n_out, int):
        n_out = [n_out]
    
    conv_input = conv2d(name+'_input', n_in, n_h, size_kernel, False, flipmask, w=w)
    
    # parameters for hidden units
    resnet = []
    for i in range(depth):
        _name = name+'_'+str(i)
        if weightsharing:
            _name = name+'[sharedw]_'+str(i)+'[/sharedw]'
        if layertype == 'a':
            resnet.append(resnet_layer_a(_name, n_h, nl, w))
        elif layertype == 'b':
            resnet.append(resnet_layer_b(_name, n_h, factor, nl, w))
        else: raise Exception()
    
    # parameters for output
    conv_out = [conv2d(name+'_out_'+str(i), n_h, n_out[i], size_kernel, True, flipmask, w=w) for i in range(len(n_out))]
    
    def f(h, h_context, w, return_hiddens=False):
        
        h = conv_input(h, w)
        if h_context != None:
            h += h_context
        
        hiddens = []
        for i in range(len(resnet)):
            h = resnet[i](h, w)
            hiddens.append(h)
        
        out = []
        for i in range(len(n_out)):
            _out = conv_out[i](h, w)
            out.append(_out)

        if len(n_out) == 1: out = out[0]
        
        if return_hiddens:
            return hiddens, out
        
        return out
    
    def postup(updates, w):
        for l in resnet: updates = l.postup(updates, w)
        for l in conv_out: updates = l.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

