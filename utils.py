import numpy as np
import glob
import re
import scipy.misc
import skimage.io
from keras.applications.vgg16 import VGG16
from keras.layers import  Input
from keras.models import  Model


ldct_filepath = r"D:\deeplearning_cai\Imaegdenoising\WGAN_LDCT\data\ldct"
ndct_filepath = r"D:\deeplearning_cai\Imaegdenoising\WGAN_LDCT\data\ndct"

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)',s)]


def load_data():
    ldct_fileList = sorted(glob.glob(ldct_filepath+"/*.jpg"),key = alphanum_key)
    ndct_fileList = sorted(glob.glob(ndct_filepath+'/*.jpg'),key = alphanum_key)
    ldct_batch = np.array([np.array(skimage.io.imread(fname).astype('float32')) for fname in ldct_fileList])
    # ldct_batch = np.array([np.array(scipy.misc.imread(fname,mode="RGB").astype('float32')) for fname in ldct_fileList])
    ndct_batch = np.array([np.array(skimage.io.imread(fname).astype('float32')) for fname in ndct_fileList])

    if ldct_batch.ndim != 3:
        ldct_batch = skimage.color.gray2rgb(ldct_batch)
        ndct_batch = skimage.color.gray2rgb(ndct_batch)

    # ndct_batch = np.array([np.array(scipy.misc.imread(fname,mode="RGB").astype('float32')) for fname in ndct_fileList])

    return ldct_batch,ndct_batch

def vgg16():
    inputs = Input((256,256,3))
    b_model = VGG16(weights = 'imagenet',include_top = False)
    model_vgg16 = Model(inputs = b_model.input,outputs = [b_model.get_layer('block1_conv2').output,
                                                          b_model.get_layer('block1_pool').output])
    return model_vgg16


def get_vggloss(target,generat):
    vgg = vgg16()
    feature_target = vgg.predict(target)
    feature_generat = vgg.predict(generat)
    w = target.shape[0]
    h = target.shape[1]
    d = target.shape[2]
    loss = np.sum(np.square(feature_target,feature_generat))
    loss_vgg = float(loss/(w*h*d))
    return loss_vgg

