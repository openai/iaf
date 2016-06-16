import numpy as np
import scipy, scipy.misc
from PIL import Image

def save_image(x, path):
    from graphy import png
    x = x.swapaxes(0, 2).swapaxes(0,1).reshape((x.shape[1],-1))
    png.from_array(x, 'RGB').save(path)

def save_raster(x, path, rescale=False, width=None):
    save_image(to_raster(x, rescale, width), path)

#def save_raster(x, path, rescale=False):
#    return Image.fromarray(to_raster(x, rescale).swapaxes(0, 2)).save(path, 'PNG')

# Shape: (n_patches,3,rows,columns)
# Or: 
def to_raster(x, rescale=False, width=None):
    #x = x.swapaxes(2, 3)
    if len(x.shape) == 3:
        x = x.reshape((x.shape[0],1,x.shape[1],x.shape[2]))
    if x.shape[1] == 1:
        x = np.repeat(x, 3, axis=1)
    if rescale:
        x = (x - x.min()) / (x.max() - x.min()) * 255.
    x = np.clip(x, 0, 255)
    assert len(x.shape) == 4
    assert x.shape[1] == 3
    n_patches = x.shape[0]
    if width is None:
        width = np.sqrt(n_patches) #result width
        assert width == int(width)
    height = n_patches/width #result height
    tile_height = x.shape[2]
    tile_width = x.shape[3]
    result = np.zeros((3,height*tile_height,width*tile_width), dtype='uint8')
    for i in range(n_patches):
        _x = (i%width)*tile_width
        y = np.floor(i/width)*tile_height
        result[:,y:y+tile_height,_x:_x+tile_width] = x[i]
    return result

    
    
