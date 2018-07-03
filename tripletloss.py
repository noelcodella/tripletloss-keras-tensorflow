# Noel C. F. Codella
# Example Triplet Loss Code for Keras / TensorFlow

# Got help from multiple web sources, including:
# 1) https://stackoverflow.com/questions/47727679/triplet-model-for-image-retrieval-from-the-keras-pretrained-network
# 2) https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/
# 3) https://keras.io/preprocessing/image/
# 4) https://github.com/keras-team/keras/issues/3386
# 5) https://github.com/keras-team/keras/issues/8130


# GLOBAL DEFINES
T_G_WIDTH = 224
T_G_HEIGHT = 224
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

# Misc. Necessities
import sys
import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imresize
np.random.seed(T_G_SEED)

# TensorFlow Includes
import tensorflow as tf
#from tensorflow.contrib.losses import metric_learning
tf.set_random_seed(T_G_SEED)

# Keras Imports & Defines 
import keras
import keras.applications
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl

from keras.preprocessing.image import ImageDataGenerator

# Generator object for data augmentation.
# Can change values here to affect augmentation style.
datagen = ImageDataGenerator(  rotation_range=90,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                )

# Local Imports
from LR_SGD import LR_SGD

# generator function for data augmentation
def createDataGen(X1, X2, X3, Y, b):

    local_seed = T_G_SEED
    genX1 = datagen.flow(X1,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX2 = datagen.flow(X2,Y, batch_size=b, seed=local_seed, shuffle=False)
    genX3 = datagen.flow(X3,Y, batch_size=b, seed=local_seed, shuffle=False)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()

            yield [X1i[0], X2i[0], X3i[0]], X1i[1]


def createModel(emb_size):

    # Initialize a ResNet50_ImageNet Model
    resnet_input = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
    resnet_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, input_tensor=resnet_input)

    # New Layers over ResNet50
    net = resnet_model.output
    net = kl.Flatten(name='flatten')(net)
    #net = kl.GlobalAveragePooling2D(name='gap')(net)
    net = kl.Dropout(0.5)(net)
    net = kl.Dense(emb_size,activation='relu',name='t_emb_1')(net)
    net = kl.Lambda(lambda  x: K.l2_normalize(x,axis=1))(net)

    # model creation
    base_model = Model(resnet_model.input, net, name="base_model")

    # triplet framework, shared weights
    input_shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS)
    input_anchor = kl.Input(shape=input_shape, name='input_anchor')
    input_positive = kl.Input(shape=input_shape, name='input_pos')
    input_negative = kl.Input(shape=input_shape, name='input_neg')

    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)

    # The Lamda layer produces output using given function. Here its Euclidean distance.
    positive_dist = kl.Lambda(euclidean_distance, name='pos_dist')([net_anchor, net_positive])
    negative_dist = kl.Lambda(euclidean_distance, name='neg_dist')([net_anchor, net_negative])

    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = kl.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist])

    model = Model([input_anchor, input_positive, input_negative], stacked_dists, name='triple_siamese')

    # Setting up optimizer designed for variable learning rate

    # Variable Learning Rate per Layers
    lr_mult_dict = {}
    for layer in resnet_model.layers:
        # comment this out to refine earlier layers
        # layer.trainable = False  
        lr_mult_dict[layer.name] = 1
    lr_mult_dict['t_emb_1'] = 100

    base_lr = 0.0001
    momentum = 0.9
    v_optimizer = LR_SGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, multipliers = lr_mult_dict)

    model.compile(optimizer=v_optimizer, loss=triplet_loss, metrics=[accuracy])

    return model


def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


# loads an image and preprocesses
def t_read_image(loc):
    t_image = cv2.imread(loc)
    t_image = imresize(t_image, (T_G_HEIGHT,T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.resnet50.preprocess_input(t_image, data_format='channels_last')

    return t_image

# loads a set of images from a text index file   
def t_read_image_list(flist, start, length):

    with open(flist) as f:
        content = f.readlines() 
    content = [x.strip().split()[0] for x in content] 

    datalen = length
    if (datalen < 0):
        datalen = len(content)

    if (start + datalen > len(content)):
        datalen = len(content) - start
 
    imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS))

    for i in range(start, start+datalen):
        if ((i-start) < len(content)):
            imgset[i-start] = t_read_image(content[i])

    return imgset


def file_numlines(fn):
    with open(fn) as f:
        return sum(1 for _ in f)


def main(argv):
    
    if len(argv) < 10:
        print 'Usage: \n\t <Train Anchors (TXT)> <Train Positives (TXT)> <Train Negatives (TXT)> <Val Anchors (TXT)> <Val Positives (TXT)> <Val Negatives (TXT)> <embedding size> <batch size> <num epochs> <output model> \n\t\tLearns triplet-loss model'
        return

    in_t_a = argv[0]
    in_t_b = argv[1]
    in_t_c = argv[2]

    in_v_a = argv[3]
    in_v_b = argv[4]
    in_v_c = argv[5]

    emb_size = int(argv[6])
    batch = int(argv[7])
    numepochs = int(argv[8])
    outpath = argv[9] 

    # chunksize is the number of images we load from disk at a time
    chunksize = batch*100
    total_t = file_numlines(in_t_a)
    total_v = file_numlines(in_v_b)
    total_t_ch = int(np.ceil(total_t / float(chunksize)))
    total_v_ch = int(np.ceil(total_v / float(chunksize)))

    print 'Dataset has ' + str(total_t) + ' training triplets, and ' + str(total_v) + ' validation triplets.'

    print 'Creating a model ...'
    model = createModel(emb_size)

    print 'Training loop ...'
    
    # manual loop over epochs to support very large sets of triplets
    for e in range(0, numepochs):

        for t in range(0, total_t_ch):

            print 'Epoch ' + str(e) + ': train chunk ' + str(t+1) + '/ ' + str(total_t_ch) + ' ...'

            print 'Reading image lists ...'
            anchors_t = t_read_image_list(in_t_a, t*chunksize, chunksize)
            positives_t = t_read_image_list(in_t_b, t*chunksize, chunksize)
            negatives_t = t_read_image_list(in_t_c, t*chunksize, chunksize)
            Y_train = np.random.randint(2, size=(1,2,anchors_t.shape[0])).T

            print 'Starting to fit ...'
            # This method does NOT use data augmentation
            # model.fit([anchors_t, positives_t, negatives_t], Y_train, epochs=numepochs,  batch_size=batch)

            # This method uses data augmentation
            model.fit_generator(generator=createDataGen(anchors_t,positives_t,negatives_t,Y_train,batch), steps_per_epoch=len(Y_train) / batch, epochs=1, shuffle=False, use_multiprocessing=True)
        
        # In case the validation images don't fit in memory, we load chunks from disk again. 
        val_res = [0.0, 0.0]
        total_w = 0.0
        for v in range(0, total_v_ch):

            print 'Loading validation image lists ...'
            print 'Epoch ' + str(e) + ': val chunk ' + str(v+1) + '/ ' + str(total_v_ch) + ' ...'
            anchors_v = t_read_image_list(in_v_a, v*chunksize, chunksize)
            positives_v = t_read_image_list(in_v_b, v*chunksize, chunksize)
            negatives_v = t_read_image_list(in_v_c, v*chunksize, chunksize)
            Y_val = np.random.randint(2, size=(1,2,anchors_v.shape[0])).T

            # Weight of current validation measurement. 
            # if loaded expected number of items, this will be 1.0, otherwise < 1.0, and > 0.0.
            w = anchors_v.shape[0] / chunksize
            total_w = total_w + w

            curval = model.evaluate([anchors_v, positives_v, negatives_v], Y_val, batch_size=batch)
            val_res[0] = val_res[0] + w*curval[0]
            val_res[1] = val_res[1] + w*curval[1]

        val_res = [x / total_w for x in val_res]

        print 'Validation Results: ' + str(val_res)

    print 'Saving model ...'
    model.save(outpath + '.h5')

    return


# Main Driver
if __name__ == "__main__":
    main(sys.argv[1:])
