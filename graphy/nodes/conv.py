
'''
Convolutional functions
'''
import numpy as np
import theano
import theano.tensor as T
if 'gpu' in theano.config.device: # @UndefinedVariable
    from theano.sandbox.cuda.dnn import dnn_conv
    from theano.sandbox.cuda.dnn import dnn_pool
elif 'cuda' in theano.config.device: # @UndefinedVariable
    from theano.sandbox.gpuarray.dnn import dnn_conv
    from theano.sandbox.gpuarray.dnn import dnn_pool
else: raise Exception()    
import graphy as G
import graphy.nodes as N

# hyperparams
logscale = True #Really works better!
bias_logscale = False
logscale_scale = 3.
init_stdev = .1
maxweight = 0.
bn = False #mean-only batchnorm

# General de-pooling inspired by Jascha Sohl-Dickstein's code
# Divides n_features by factor**2, multiplies width/height factor 
def depool2d_split(x, factor=2):
    assert factor >= 1
    if factor == 1: return x
    #assert x.shape[1] >= 4 and x.shape[1]%4 == 0
    x = x.reshape((x.shape[0], x.shape[1]/factor**2, factor, factor, x.shape[2], x.shape[3]))
    x = x.dimshuffle(0, 1, 4, 2, 5, 3)
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2]*x.shape[3], x.shape[4]*x.shape[5]))
    return x

# General nearest-neighbour downsampling inspired by Jascha Sohl-Dickstein's code
def downsample2d_nearest_neighbour(x, scale=2):
    x = x.reshape((x.shape[0], x.shape[1], x.shape[2]/scale, scale, x.shape[3]/scale, scale))
    x = T.mean(x, axis=5)
    x = T.mean(x, axis=3)
    return x

# 2X nearest-neighbour upsampling, also inspired by Jascha Sohl-Dickstein's code
def upsample2d_nearest_neighbour(x):
    shape = x.shape
    x = x.reshape((shape[0], shape[1], shape[2], 1, shape[3], 1))
    x = T.concatenate((x, x), axis=5)
    x = T.concatenate((x, x), axis=3)
    x = x.reshape((shape[0], shape[1], shape[2]*2, shape[3]*2))
    return x 

# 2X nearest-neighbour upsampling, also inspired by Jascha Sohl-Dickstein's code
def upsample2d_perforated(x):
    shape = x.shape
    x = x.reshape((shape[0], shape[1], shape[2], 1, shape[3], 1))
    y = T.zeros((shape[0], shape[1], shape[2], 2, shape[3], 2),dtype=G.floatX)
    x = T.set_subtensor(y[:,:,:,0:1,:,0:1], x)
    x = x.reshape((shape[0], shape[1], shape[2]*2, shape[3]*2))
    return x 

# Pad input
def pad2d(x, n_padding):
    result_shape = (x.shape[0],x.shape[1],x.shape[2]+2*n_padding,x.shape[3]+2*n_padding)
    result = T.zeros(result_shape, dtype=G.floatX)
    result = T.set_subtensor(result[:,:,n_padding:-n_padding,n_padding:-n_padding], x)
    return result


# Pad input, add extra channel
def pad2dwithchannel(x, size_kernel):
    assert size_kernel[0]>1 or size_kernel[1]>1
    assert size_kernel[0]%2 == 1
    assert size_kernel[1]%2 == 1
    a = (size_kernel[0]-1)/2
    b = (size_kernel[1]-1)/2
    if True:
        n_channels = x.shape[1]
        result_shape = (x.shape[0],x.shape[1]+1,x.shape[2]+2*a,x.shape[3]+2*b)
        result = T.zeros(result_shape, dtype=G.floatX)
        result = T.set_subtensor(result[:,n_channels,:,:], 1.)
        result = T.set_subtensor(result[:,n_channels,a:-a,b:-b], 0.)
        result = T.set_subtensor(result[:,:n_channels,a:-a,b:-b], x)
    else:
        # new code, requires that the minibatch size 'x.tag.test_value.shape[0]' is the same during execution
        # I thought this would be more memory-efficient, but seems not the case in practice
        print 'new code, requires that the minibatch size "x.tag.test_value.shape[0]" is the same during execution' 
        x_shape = x.tag.test_value.shape
        n_channels = x_shape[1]
        result_shape = (x_shape[0],x_shape[1]+1,x_shape[2]+2*a,x_shape[3]+2*b)
        result = np.zeros(result_shape,dtype=G.floatX)
        result[:,n_channels,:,:] = 1.
        result[:,n_channels,a:-a,b:-b] = 0.
        result = T.constant(result)
        result = T.set_subtensor(result[:,:n_channels,a:-a,b:-b], x)
    return result


# Multi-scale conv
def msconv2d(name, n_scales, n_in, n_out, size_kernel=(3,3), pad_channel=True, border_mode='valid', downsample=1, upsample=1, w={}):
    convs = [conv2d(name+"_s"+str(i), n_in, n_out, size_kernel, pad_channel, border_mode, downsample, upsample, w) for i in range(n_scales)]
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

# 2D conv with input bias
# size_kernel = (n_rows, n_cols)
def conv2d(name, n_in, n_out, size_kernel=(3,3), pad_channel=True, border_mode='valid', downsample=1, upsample=1, datainit=True, zeroinit=False, l2norm=True,  w={}):
    
    # TODO FIX: blows up parameters if all inputs are 0
    
    if not pad_channel:
        border_mode = 'same'
        print 'No pad_channel, changing border_mode to same'

    if '[sharedw]' in name and '[/sharedw]' in name:
        name_w = name
        pre, b = name.split("[sharedw]")
        number, post = b.split("[/sharedw]")
        name_w = pre+"[s]"+post
        name = pre+number+post # Don't share the bias and scales
        #name = name_w # Also share the bias and scales
    else:
        name_w = name
    
    if type(downsample) == int:
        downsample = (downsample,downsample)
    assert type(downsample) == tuple
    assert border_mode in ['valid','full','same']
    
    _n_in = n_in
    _n_out = n_out
    if upsample > 1:
        _n_out = n_out * upsample**2
    
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
    
    def l2normalize(kerns):
        norm = T.sqrt((kerns**2).sum(axis=(1,2,3), keepdims=True))
        return kerns / norm
    def maxconstraint(kerns):
        return kerns * (maxweight / T.maximum(maxweight, abs(kerns).max(axis=(1,2,3), keepdims=True)))

    if zeroinit:
        w[name_w+'_w'] = G.sharedf(np.zeros((_n_out, _n_in, size_kernel[0], size_kernel[1])))
        datainit = False
    else: 
        w[name_w+'_w'] = G.sharedf(0.05*np.random.randn(_n_out, _n_in, size_kernel[0], size_kernel[1]))
        if maxweight > 0:
            w[name_w+'_w'].set_value(maxconstraint(w[name_w+'_w']).tag.test_value)
    
    w[name+'_b'] = G.sharedf(np.zeros((_n_out,)))
    if bias_logscale:
        w[name+'_bs'] = G.sharedf(0.)
    
    if l2norm:
        if logscale:
            w[name+'_s'] = G.sharedf(np.zeros((_n_out,)))
        else:
            w[name+'_s'] = G.sharedf(np.ones((_n_out,)))
    elif do_constant_rescale:
        print 'WARNING: constant rescale, these weights arent saved'
        constant_rescale = G.sharedf(np.ones((_n_out,)))
    
    
    def f(h, w):
        
        input_shape = h.tag.test_value.shape[1:]

        _input = h
        
        if pad_channel:
            h = pad2dwithchannel(h, size_kernel)

        kerns = w[name_w+'_w']
        #if name == '1_down_conv1':
        #    kerns = T.printing.Print('kerns 1')(kerns)
        if l2norm:
            kerns = l2normalize(kerns)
            if logscale:
                kerns *= T.exp(logscale_scale*w[name+'_s']).dimshuffle(0,'x','x','x')
            else:
                kerns *= w[name+'_s'].dimshuffle(0,'x','x','x')
        elif do_constant_rescale:
            kerns *= constant_rescale.dimshuffle(0,'x','x','x')
        
        #if name == '1_down_conv1':
        #    kerns = T.printing.Print('kerns 2')(kerns)
        
        h = dnn_conv(h, kerns, border_mode=border_mode, subsample=downsample)

        # Mean-only batch norm
        if bn: 
            h -= h.mean(axis=(0,2,3), keepdims=True)
        
        _b = w[name+'_b'].dimshuffle('x',0,'x','x')
        if bias_logscale:
            _b *= T.exp(logscale_scale * w[name+'_bs'])
        h += _b
        
        if '__init' in w and datainit:
            
            # Std
            data_std = h.std(axis=(0,2,3))
            num_zeros = (data_std.tag.test_value == 0).sum()
            if num_zeros > 0:
                print "Warning: Stdev=0 for "+str(num_zeros)+" features in "+name+". Skipping data-dependent init."
            else:
                
                std = (1./init_stdev) * data_std
                std += 1e-7
                
                if name+'_s' in w:
                    if logscale:
                        w[name+'_s'].set_value(-T.log(std).tag.test_value/logscale_scale)
                    else:
                        w[name+'_s'].set_value((1./std).tag.test_value)
                elif do_constant_rescale:
                    constant_rescale.set_value((1./std).tag.test_value)
                
                h /= std.dimshuffle('x',0,'x','x')
                
                # Mean
                mean = h.mean(axis=(0,2,3))
                w[name+'_b'].set_value(-mean.tag.test_value)
                h -= mean.dimshuffle('x',0,'x','x')
            
                #print name, w[name+'_w'].get_value().mean(), w[name+'_w'].get_value().std(), w[name+'_w'].get_value().max()
        
        if upsample>1:
            h = depool2d_split(h, factor=upsample)
        
        if not '__init' in w:
            output_shape = h.tag.test_value.shape[1:]
            print 'conv2d', name, input_shape, output_shape, size_kernel, pad_channel, border_mode, downsample, upsample
        
        #print name, abs(h).max().tag.test_value, abs(h).min().tag.test_value
        #h = T.printing.Print(name)(h)
        
        return h
    
    # Normalize weights to _norm L2 norm
    # TODO: check whether only_upper_bounds here really helps
    # (the effect is a higher learning rate in the beginning of training)
    def postup(updates, w):
        if l2norm and maxweight>0.:
            updates[w[name_w+'_w']] = maxconstraint(updates[w[name_w+'_w']])
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# ResNet layer
def resnetv1_layer(name, n_in, n_out, size_kernel=(3,3), downsample=1, upsample=1, nl='relu', w={}):
    #print 'resnet_layer', name, shape_in, shape_out, size_kernel, downsample, upsample
    
    f_nl = N.nonlinearity(name+"_nl", nl)
    
    border_mode = 'valid'
    
    if upsample == 1:
        # either no change in shape, or subsampling
        conv1 = conv2d(name+'_conv1', n_in, n_out, size_kernel, True, border_mode, downsample, upsample, w=w)
        conv2 = conv2d(name+'_conv2', n_out, n_out, size_kernel, True, border_mode, downsample=1, upsample=1, w=w)
        conv3 = None
        if downsample>1 or upsample>1 or n_out != n_in:
            conv3 = conv2d(name+'_conv3', n_in, n_out, (downsample, downsample), None, 'valid', downsample, upsample, w=w)
    else:
        # upsampling
        assert downsample == 1
        conv1 = conv2d(name+'_conv1', n_in, n_in, size_kernel, True, border_mode, downsample=1, upsample=1, w=w)
        conv2 = conv2d(name+'_conv2', n_in, n_out, size_kernel, True, border_mode, downsample, upsample, w=w)
        conv3 = None
        if downsample>1 or upsample>1 or n_out != n_in:
            conv3 = conv2d(name+'_conv3', n_in, n_out, (downsample, downsample), None, 'valid', downsample, upsample, w=w)
    
    def f(_input, w):
        hidden = f_nl(conv1(_input, w))
        _output = .1 * conv2(hidden, w)
        if conv3 != None:
            return T.nnet.relu(conv3(_input, w) + _output)
        return T.nnet.relu(_input + _output)
    
    def postup(updates, w):
        updates = conv1.postup(updates, w)
        updates = conv2.postup(updates, w)
        if conv3 != None:
            updates = conv3.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# ResNet v1 with n_layers layers
# Support sub/upsampling
# In case of subsampling, first layer does subsampling (like in the ResNet paper)
# In case of upsampling, the last layer does the upsampling (to make the net symmetrical)
def resnetv1(name, n_layers, n_in, n_out, size_kernel=(3,3), downsample=1, upsample=1, nl='relu', w={}):
    layers = []
    for i in range(n_layers):
        _n_in = n_in
        _n_out = n_out
        _downsample = downsample
        _upsample = upsample
        if _downsample > 1 and i > 0:
            _downsample = 1
            _n_in = n_out
        if _upsample > 1 and i < n_layers-1:
            _upsample = 1
            _n_out = n_in
        
        layer = resnetv1_layer(name+'_'+str(i), _n_in, _n_out, size_kernel, _downsample, _upsample, nl, w)
        layers.append(layer)
    
    def f(h, w):
        for i in range(n_layers):
            h = layers[i](h, w)
        return h
    
    def postup(updates, w):
        for i in range(n_layers):
            updates = layers[i].postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)
    



# ResNet V2 layer
def resnetv2_layer_a(name, n_feats, nl='relu', w={}):

    f_nl = N.nonlinearity(name+"_nl", nl)
    
    # either no change in shape, or subsampling
    conv1 = conv2d(name+'_conv1', n_feats, n_feats, (3,3), w=w)
    conv2 = conv2d(name+'_conv2', n_feats, n_feats, (3,3), w=w)
    
    def f(_input, w):
        h = _input
        h = f_nl(conv1(h, w))
        h = conv2(h, w)
        return T.nnet.relu(_input + .1 * h)
    
    def postup(updates, w):
        updates = conv1.postup(updates, w)
        updates = conv2.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# ResNet V2 layer
def resnetv2_layer_b(name, n_feats, factor=4, nl='relu', w={}):

    f_nl = N.nonlinearity(name+"_nl", nl)
    
    # either no change in shape, or subsampling
    conv1 = conv2d(name+'_conv1', n_feats, n_feats/factor, (1,1), w=w)
    conv2 = conv2d(name+'_conv2', n_feats/factor, n_feats/factor, (3,3), w=w)
    conv3 = conv2d(name+'_conv3', n_feats/factor, n_feats, (1,1), w=w)
    
    def f(_input, w):
        h = _input
        h = f_nl(conv1(h, w))
        h = f_nl(conv2(h, w))
        h = conv3(h, w)
        return T.nnet.relu(_input + .1 * h)
    
    def postup(updates, w):
        updates = conv1.postup(updates, w)
        updates = conv2.postup(updates, w)
        updates = conv3.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)


# ResNet V2 with n_layers layers
# V2: no sub/upsampling, not changing nr of features, bottleneck layer, fixed kernel size (1x1 and 3x3)
def resnetv2(name, n_layers, n_feats, layertype='a', factor=4, nl='relu', w={}):
    
    layers = []
    for i in range(n_layers):
        if layertype == 'a':
            layers.append(resnetv2_layer_a(name+'_'+str(i), n_feats, nl, w))
        if layertype == 'b':
            layers.append(resnetv2_layer_b(name+'_'+str(i), n_feats, factor, nl, w))
    
    def f(h, w):
        for i in range(n_layers):
            h = layers[i](h, w)
        return h
    
    def postup(updates, w):
        for i in range(n_layers):
            updates = layers[i].postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# ResNet V3 layer
def resnetv3_layer_a(name, n_feats, nl='softplus', alpha=.1, w={}):
    
    f_nl1 = N.nonlinearity(name+"_nl1", nl)
    f_nl2 = N.nonlinearity(name+"_nl2", nl)
    
    # either no change in shape, or subsampling
    conv1 = conv2d(name+'_conv1', n_feats, n_feats, (3,3), w=w)
    conv2 = conv2d(name+'_conv2', n_feats, n_feats, (3,3), w=w)
    
    def f(_input, w):
        h = f_nl1(_input)
        h = f_nl2(conv1(h, w))
        h = conv2(h, w)
        return _input + alpha * h
    
    def postup(updates, w):
        updates = conv1.postup(updates, w)
        updates = conv2.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# ResNet V3 layer
def resnetv3_layer_b(name, n_feats, factor=4, nl='softplus', alpha=.1, w={}):

    f_nl1 = N.nonlinearity(name+"_nl1", nl)
    f_nl2 = N.nonlinearity(name+"_nl2", nl)
    f_nl3 = N.nonlinearity(name+"_nl3", nl)
    
    # either no change in shape, or subsampling
    conv1 = conv2d(name+'_conv1', n_feats, n_feats/factor, (1,1), w=w)
    conv2 = conv2d(name+'_conv2', n_feats/factor, n_feats/factor, (3,3), w=w)
    conv3 = conv2d(name+'_conv3', n_feats/factor, n_feats, (1,1), w=w)
    
    def f(_input, w):
        h = f_nl1(_input)
        h = f_nl2(conv1(h, w))
        h = f_nl3(conv2(h, w))
        h = conv3(h, w)
        return _input + alpha * h
        
    def postup(updates, w):
        updates = conv1.postup(updates, w)
        updates = conv2.postup(updates, w)
        updates = conv3.postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)

# ResNet V3 with n_layers layers
# V3: like V2, but nonlinearity applied in more logical manner: as first element of inner functions
def resnetv3(name, n_layers, n_feats, nl='softplus', layertype='a', factor=4, w={}):
    
    layers = []
    for i in range(n_layers):
        if layertype == 'a':
            layers.append(resnetv3_layer_a(name+'_'+str(i), n_feats, nl, .1/n_layers, w))
        if layertype == 'b':
            layers.append(resnetv3_layer_b(name+'_'+str(i), n_feats, factor, nl, .1/n_layers, w))
    
    def f(h, w):
        for i in range(n_layers):
            h = layers[i](h, w)
        return h
    
    def postup(updates, w):
        for i in range(n_layers):
            updates = layers[i].postup(updates, w)
        return updates
    
    return G.Struct(__call__=f, w=w, postup=postup)
