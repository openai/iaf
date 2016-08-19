import graphy as G
import numpy as np
import time, sys, os
from sacred import Experiment
from __builtin__ import False

ex = Experiment('Deep VAE')

@ex.config
def config():
    
    # optimization:
    n_reporting = 10 #epochs between reporting
    px = 'logistic'
    pad_x = 0
    
    # datatype
    problem = 'cifar10'
    n_batch = 16 # Minibatch size
    if problem == 'mnist':
        shape_x = (1,28,28)
        px = 'bernoulli'
        pad_x = 2
        n_h = 64
        n_z = 32
    if problem == 'cifar10':
        shape_x = (3,32,32)
        n_h = 160
        n_z = 32
    if problem == 'svhn':
        shape_x = (3,32,32)
        n_reporting = 1
        n_h = 160
        n_z = 32
    if problem == 'lfw':
        shape_x = (3,64,48)
        n_h = 160
        n_z = 32

    n_h1 = n_h
    n_h2 = n_h

    # dataset
    n_train = 0
    
    # model
    model_type = 'cvae1'
        
    if model_type == 'cvae1':
        depths = [2,2]
        
        margs = {
            'shape_x': shape_x,
            'depths': depths,
            'n_h1': n_h1,
            'n_h2': n_h2,
            'n_z': n_z,
            'prior': 'diag',
            'posterior': 'down_diag',
            'px': px,
            'nl': 'elu',
            'kernel_x': (5,5),
            'kernel_h': (3,3),
            'kl_min': 0.25,
            'optim': 'adamax',
            'alpha': 0.002,
            'beta1': 0.1,
            'pad_x': pad_x,
            'weightsharing': False,
            'depth_ar': 1,
            'downsample_type': 'nn'
        }
    
    if model_type == 'simplecvae1':
        depths = [2,2,2]
        widths = [32,64,128]
        
        margs = {
            'shape_x': shape_x,
            'depths': depths,
            'widths': widths,
            'n_z': n_z,
            'prior': 'diag',
            'posterior': 'down_diag',
            'px': px,
            'nl': 'elu',
            'kernel_x': (5,5),
            'kernel_h': (3,3),
            'kl_min': 0.25,
            'optim': 'adamax',
            'alpha': 0.002,
            'beta1': 0.1,
            'pad_x': pad_x,
            'weightsharing': False
        }
    
    # model loading/saving
    save_model = True
    load_model_path = None
    load_model_complete = True # Whether loaded parameters are complete
    
    # Estimate the marginal likelihood
    est_marglik = 0.
    est_marglik_data = 'valid'
    
def init_logs():
    global logpath, logdir
    # Create log directory
    logdir = str(time.time())
    logpath = os.environ['ML_LOG_PATH']+'/'+logdir+'/'
    print 'Logpath: '+logpath
    os.makedirs(logpath)
    # Log stdout messages to file
    sys.stdout = G.misc.logger.Logger(logpath+"log.txt")
    # Clone local source to logdir
    os.system("rsync -au --include '*/' --include '*.py' --exclude '*' . "+logpath+"source")
    with open(logpath+"source/run.sh", 'w') as f:
        f.write("python "+" ".join(sys.argv)+"\n")
    os.chmod(logpath+"source/run.sh", 0700)
    
@ex.capture
def construct_model(data_init, model_type, margs, load_model_path, load_model_complete, n_batch):
    import models
    margs['data_init'] = data_init
    if model_type == 'fcvae1':
        model = models.fcvae(**margs)
    if model_type == 'cvae1':
        model = models.cvae1(**margs)
    if model_type == 'simplecvae1':
        import simplemodel
        model = simplemodel.simplecvae1(**margs)
    
    if load_model_path != None:
        print 'Loading existing model at '+load_model_path
        _w = G.ndict.np_loadz(load_model_path+'/weights.ndict.tar.gz')
                
        G.ndict.set_value(model.w, _w, load_model_complete)
        G.ndict.set_value(model.w_avg, _w, load_model_complete)
    
    return model

@ex.capture
def get_data(problem, n_train, n_batch):
        
    if problem == 'cifar10':
        # Load data
        data_train, data_valid = G.misc.data.cifar10(False)
    if problem == 'svhn':
        # Load data
        data_train, data_valid = G.misc.data.svhn(False, True)
    elif problem == 'mnist':
        # Load data
        validset = False
        if validset:
            data_train, data_valid, data_test = G.misc.data.mnist_binarized(validset, False)
        else:
            data_train, data_valid = G.misc.data.mnist_binarized(validset, False)
        data_train['x'] = data_train['x'].reshape((-1,1,28,28))
        data_valid['x'] = data_valid['x'].reshape((-1,1,28,28))
    elif problem == 'lfw':
        data_train = G.misc.data.lfw(False,True)
        data_valid = G.ndict.getRows(data_train, 0, 1000)
    

    data_init = {'x':data_train['x'][:n_batch]}
        
    if n_train > 0:
        data_train = G.ndict.getRows(data_train, 0, n_train)
        data_valid = G.ndict.getRows(data_valid, 0, n_train)
    
    return data_train, data_valid, data_init

@ex.automain
def train(shape_x, problem, n_batch, n_train, n_reporting, save_model, est_marglik, est_marglik_data, margs):
    
    global logpath
    
    # Initialize logs
    init_logs()
    
    # Get data
    data_train, data_valid, data_init = get_data()
    
    # Construct model
    model = construct_model(data_init)
    
    # Estimate the marginal likelihood
    if est_marglik > 0:
        if est_marglik_data == 'valid':
            data = data_valid
        elif est_marglik_data == 'train':
            data = data_train
        # Correction since model's actual cost is divided by this factor
        correctionfactor = - (np.prod(shape_x) * np.log(2.))
        obj_test = []
        for i in range(est_marglik):
            cost = model.eval(data, n_batch=n_batch, randomorder=False)['cost'] * correctionfactor
            obj_test.append(cost)
            _obj = np.vstack(obj_test)
            _max = np.max(_obj, axis=0)
            _est = np.log(np.exp(_obj - _max).mean(axis=0)) + _max
            if i%1 == 0:
                print 'Estimate of logp(x) after', i+1, 'samples:', _est.mean() / correctionfactor
        raise Exception()
        sys.exit()
    
    # Report
    cost_best = [None]
    eps_fixed = model.eps({'n_batch':100})
    def report(epoch, dt, cost):
        if np.isnan(cost):
            raise Exception('NaN detected!!')
        
        results_valid = model.eval(data_valid, n_batch=n_batch)
        for i in results_valid: results_valid[i] = results_valid[i].mean()
        
        _w = G.ndict.get_value(model.w_avg)
        G.ndict.np_savez(_w, logpath+'weights')
        
        if cost_best[0] is None or results_valid['cost'] < cost_best[0]:
            cost_best[0] = results_valid['cost']
            if save_model:
                G.ndict.np_savez(_w, logpath+'weights_best')
        
        if True:
            # Write all results to file
            with open(logpath+"results.txt", "a") as log:
                if epoch == 0:
                    log.write("Epoch "+" ".join(map(str, results_valid.keys())) + "\n")
                log.write(str(epoch)+" "+" ".join(map(str, results_valid.values())) + "\n")
        
        if True:
            eps = model.eps({'n_batch':100})
            image = model.decode(eps)
            G.graphics.save_raster(image, logpath+'sample_'+str(epoch)+'.png')
            image = model.decode(eps_fixed)
            G.graphics.save_raster(image, logpath+'sample_fixed1_'+str(epoch)+'.png')
            
            #eps_fixed_copy = G.ndict.clone(eps_fixed)
            #for i in range(len(eps_fixed)):
            #    eps_fixed_copy['']
        
        if epoch == 0:
            print 'logdir:', 't:', 'Epoch:', 'Train cost:', 'Valid cost:', 'Best:', 'log(stdev) of p(x|z):'
        
        logsd_x = 0.
        if 'logsd_x' in model.w_avg:
            logsd_x = model.w_avg['logsd_x'].get_value()
        print logdir, '%.2f'%dt, epoch, '%.5f'%cost, '%.5f'%results_valid['cost'], '%.5f'%cost_best[0], logsd_x
    
    print 'Training'
    
    for epoch in xrange(1000000):
        t0 = time.time()
        
        result = model.train(data_train, n_batch=n_batch)
        
        if epoch <= 10 or epoch%n_reporting == 0:
            report(epoch, time.time()-t0, cost=np.mean(result))
    

