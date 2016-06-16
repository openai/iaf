import os
import numpy as np
import math
import graphy as G

basepath = os.environ['ML_DATA_PATH']

''' Standard datasets
The first dimension of tensors goes over datapoints.
'''

def mnist(with_y=True):

    n_y = 10
    import scipy.io
    data = scipy.io.loadmat(basepath+'/mnist_roweis/mnist_all.mat')
    train_x = [data['train'+str(i)] for i in range(n_y)]
    train_y = [(i*np.ones((train_x[i].shape[0],))).astype(np.uint8) for i in range(n_y)]
    test_x = [data['test'+str(i)] for i in range(n_y)]
    test_y = [(i*np.ones((test_x[i].shape[0],))).astype(np.uint8) for i in range(n_y)]
    
    train = {'x':np.concatenate(train_x)}
    test = {'x':np.concatenate(test_x)}

    if with_y:
        train['y'] = np.concatenate(train_y)
        test['y'] = np.concatenate(test_y)
    
    G.ndict.shuffle(train) #important!!
    G.ndict.shuffle(test) #important!!
    return train, test
    
'''
Binarized MNIST (by Hugo Larochelle)
'''
def mnist_binarized(validset=False, flattened=True):
    path = basepath+'/mnist_binarized/'
    import h5py
    train = {'x':h5py.File(path+"binarized_mnist-train.h5")['data'][:].astype('uint8')*255}
    valid = {'x':h5py.File(path+"binarized_mnist-valid.h5")['data'][:].astype('uint8')*255}
    test = {'x':h5py.File(path+"binarized_mnist-test.h5")['data'][:].astype('uint8')*255}
    G.ndict.shuffle(train)
    G.ndict.shuffle(test)
    G.ndict.shuffle(valid)
    if not flattened:
        for data in [train,valid,test]:
            data['x'] = data['x'].reshape((-1,1,28,28))
    if not validset:
        print "Full training set"
        train['x'] = np.concatenate((train['x'], valid['x']))
        return train, test
    return train, valid, test
    
# Converts integer labels to binarized labels (1-of-K coding)
def binarize_labels(y, n_classes=10):
    new_y = np.zeros((y.shape[0], n_classes), dtype=G.floatX)
    for i in range(y.shape[0]):
        new_y[i, y[i]] = 1
    return new_y

'''
Create semi-supervised sets of labeled and unlabeled data
where there are equal number of labels from each class
x: dict with dataset
key_y: name (key) of label variable in x
shuffle: whether to shuffle the input and output
n_labeled: number of labeled instances
'''
def create_semisupervised(x, key_y, n_labeled, shuffle=True):
    if shuffle:
        G.ndict.shuffle(x)
    n_classes = np.amax(x[key_y])+1
    if n_labeled%n_classes != 0: raise("Cannot create stratisfied semi-supervised set since n_labeled (wished number of labeled samples) not divisible by n_classes (number of classes)")
    n_labels_per_class = n_labeled/n_classes
    x_l = {j: [0]*n_classes for j in x} #labeled
    x_u = {j: [0]*n_classes for j in x} #unlabeld
    for i in range(n_classes):
        idx = x[key_y] == i
        for j in x:
            x_l[j][i] = x[j][idx][:n_labels_per_class]
            x_u[j][i] = x[j][idx][n_labels_per_class:]
    x_l = {i: np.concatenate(x_l[i]) for i in x}
    x_u = {i: np.concatenate(x_u[i]) for i in x}
    if shuffle:
        G.ndict.shuffle(x_l)
        G.ndict.shuffle(x_u)
    return x_l, x_u


# from http://cs.nyu.edu/~roweis/data.html
# returned pixels are uint8
def cifar10(with_y=True, binarize_y=False):
    # Load the original images into numpy arrays
    def unpickle(file):
        import cPickle
        fo = open(file, 'rb')
        result = cPickle.load(fo)
        fo.close()
        return result
    path = os.environ['CIFAR10_PATH']
    n_train = 5
    _train = [unpickle(path+'data_batch_'+str(i+1)) for i in range(n_train)]
    train = {'x':np.concatenate([_train[i]['data'] for i in range(n_train)])}
    _test = unpickle(path+'test_batch')
    test = {'x':_test['data']}
    
    train['x'] = train['x'].reshape((-1,3,32,32))
    test['x'] = test['x'].reshape((-1,3,32,32))
    
    if with_y:
        train['y'] = np.concatenate([_train[i]['labels'] for i in range(n_train)])
        test['y'] = np.asarray(_test['labels'])
        if binarize_y:
            train['y'] = binarize_labels(train['y'])
            test['y'] = binarize_labels(test['y'])
    
    G.ndict.shuffle(train)
    G.ndict.shuffle(test)
    return train, test

# SVHN data
def svhn(with_y=True, with_extra=False, binarize_y=False):
    path = os.environ['ML_DATA_PATH']+'/svhn'
    import scipy.io
    train = scipy.io.loadmat(path+'/train_32x32.mat')
    train_x = train['X'].transpose((3,2,0,1))
    if with_extra:
        assert not with_y
        extra_x = scipy.io.loadmat(path+'_extra/extra_32x32.mat')['X'].transpose((3,2,0,1))
        train_x = np.concatenate((train_x,extra_x),axis=0)
    
    test = scipy.io.loadmat(path+'/test_32x32.mat')
    test_x = test['X'].transpose((3,2,0,1))
    
    if with_y:
        train_y = train['y'].reshape((-1,)) - 1
        test_y = test['y'].reshape((-1,)) - 1
        if binarize_y:
            train['y'] = binarize_labels(train['y'])
            test['y'] = binarize_labels(test['y'])
        return {'x':train_x, 'y':train_y}, {'x':test_x, 'y':test_y}
    
    return {'x':train_x}, {'x':test_x}

# SVHN data
def lfw(with_y=True, pad=False):
    path = os.environ['ML_DATA_PATH']+'/lfw/'
    data = {'x':np.load(path+'lfw_62x47.npy').transpose((0,3,1,2))}
    if pad:
        padded = np.zeros((data['x'].shape[0],3,64,48),dtype='uint8')
        padded[:,:,:-2,:-1] = data['x']
        data['x'] = padded
        
    if with_y:
        data['y'] = np.load(path+'lfw_labels.npy')
    return data
