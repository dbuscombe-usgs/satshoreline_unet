# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from imports import *
import tensorflow.keras.backend as K
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk

for BATCH_SIZE in [4,6,8,12]:

    print(BATCH_SIZE)

    if BATCH_SIZE<=12:
        min_lr = start_lr = 1e-5
        max_lr = 1e-4
        rampup_epochs = 20
    else:
        min_lr = start_lr = 1e-7
        max_lr = 1e-5
        rampup_epochs = 20

    #-----------------------------------
    def crf_refine(label, img, theta_col=100, theta_spat=3):
        """
        "crf_refine(label, img)"
        This function refines a label image based on an input label image and the associated image
        Uses a conditional random field algorithm using spatial and image features
        INPUTS:
            * label [ndarray]: label image 2D matrix of integers
            * image [ndarray]: image 3D matrix of integers
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: None
        OUTPUTS: label [ndarray]: label image 2D matrix of integers
        """

        nclasses = 4#5
        H = label.shape[0]
        W = label.shape[1]
        U = unary_from_labels(1+label,nclasses,gt_prob=0.51)
        d = dcrf.DenseCRF2D(H, W, nclasses)
        d.setUnaryEnergy(U)

        # to add the color-independent term, where features are the locations only:
        d.addPairwiseGaussian(sxy=(theta_spat, theta_spat),
                     compat=3,
                     kernel=dcrf.DIAG_KERNEL,
                     normalization=dcrf.NORMALIZE_SYMMETRIC)
        feats = create_pairwise_bilateral(
                              sdims=(theta_col, theta_col),
                              schan=(2,2,2),
                              img=img,
                              chdim=2)

        d.addPairwiseEnergy(feats, compat=120,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(20)
        kl1 = d.klDivergence(Q)
        return np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8), kl1


    ###############################################################
    ### MODEL FUNCTIONS
    ###############################################################
    #-----------------------------------
    def batchnorm_act(x):
        """
        batchnorm_act(x)
        This function applies batch normalization to a keras model layer, `x`, then a relu activation function
        INPUTS:
            * `z` : keras model layer (should be the output of a convolution or an input layer)
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: None
        OUTPUTS:
            * batch normalized and relu-activated `x`
        """
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation("relu")(x)

    #-----------------------------------
    def conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
        """
        conv_block(x, filters, kernel_size = (7,7), padding="same", strides=1)
        This function applies batch normalization to an input layer, then convolves with a 2D convol layer
        The two actions combined is called a convolutional block

        INPUTS:
            * `filters`: number of filters in the convolutional block
            * `x`:input keras layer to be convolved by the block
        OPTIONAL INPUTS:
            * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
            * `padding`="same":  see tf.keras.layers.Conv2D
            * `strides`=1: see tf.keras.layers.Conv2D
        GLOBAL INPUTS: None
        OUTPUTS:
            * keras layer, output of the batch normalized convolution
        """
        conv = batchnorm_act(x)
        return tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

    #-----------------------------------
    def bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
        """
        bottleneck_block(x, filters, kernel_size = (7,7), padding="same", strides=1)

        This function creates a bottleneck block layer, which is the addition of a convolution block and a batch normalized/activated block
        INPUTS:
            * `filters`: number of filters in the convolutional block
            * `x`: input keras layer
        OPTIONAL INPUTS:
            * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
            * `padding`="same":  see tf.keras.layers.Conv2D
            * `strides`=1: see tf.keras.layers.Conv2D
        GLOBAL INPUTS: None
        OUTPUTS:
            * keras layer, output of the addition between convolutional and bottleneck layers
        """
        conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
        conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

        bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        bottleneck = batchnorm_act(bottleneck)

        return tf.keras.layers.Add()([conv, bottleneck])

    #-----------------------------------
    def res_block(x, filters, kernel_size = (7,7), padding="same", strides=1):
        """
        res_block(x, filters, kernel_size = (7,7), padding="same", strides=1)

        This function creates a residual block layer, which is the addition of a residual convolution block and a batch normalized/activated block
        INPUTS:
            * `filters`: number of filters in the convolutional block
            * `x`: input keras layer
        OPTIONAL INPUTS:
            * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
            * `padding`="same":  see tf.keras.layers.Conv2D
            * `strides`=1: see tf.keras.layers.Conv2D
        GLOBAL INPUTS: None
        OUTPUTS:
            * keras layer, output of the addition between residual convolutional and bottleneck layers
        """
        res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
        res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

        bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
        bottleneck = batchnorm_act(bottleneck)

        return tf.keras.layers.Add()([bottleneck, res])

    #-----------------------------------
    def upsamp_concat_block(x, xskip):
        """
        upsamp_concat_block(x, xskip)
        This function takes an input layer and creates a concatenation of an upsampled version and a residual or 'skip' connection
        INPUTS:
            * `xskip`: input keras layer (skip connection)
            * `x`: input keras layer
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: None
        OUTPUTS:
            * keras layer, output of the addition between residual convolutional and bottleneck layers
        """
        u = tf.keras.layers.UpSampling2D((2, 2))(x)
        return tf.keras.layers.Concatenate()([u, xskip])

    #-----------------------------------
    def iou(obs, est, nclasses):
        IOU=0
        for n in range(1,nclasses):
            component1 = obs==n
            component2 = est==n
            overlap = component1*component2 # Logical AND
            union = component1 + component2 # Logical OR
            calc = overlap.sum()/float(union.sum())
            if not np.isnan(calc):
                IOU += calc
            if IOU>1:
                IOU=IOU/n
        return IOU

    #-----------------------------------
    def res_unet(sz, f, nclasses=1):
        """
        res_unet(sz, f, nclasses=1)
        This function creates a custom residual U-Net model for image segmentation
        INPUTS:
            * `sz`: [tuple] size of input image
            * `f`: [int] number of filters in the convolutional block
            * flag: [string] if 'binary', the model will expect 2D masks and uses sigmoid. If 'multiclass', the model will expect 3D masks and uses softmax
            * nclasses [int]: number of classes
        OPTIONAL INPUTS:
            * `kernel_size`=(3, 3): tuple of kernel size (x, y) - this is the size in pixels of the kernel to be convolved with the image
            * `padding`="same":  see tf.keras.layers.Conv2D
            * `strides`=1: see tf.keras.layers.Conv2D
        GLOBAL INPUTS: None
        OUTPUTS:
            * keras model
        """
        inputs = tf.keras.layers.Input(sz)

        ## downsample
        e1 = bottleneck_block(inputs, f); f = int(f*2)
        e2 = res_block(e1, f, strides=2); f = int(f*2)
        e3 = res_block(e2, f, strides=2); f = int(f*2)
        e4 = res_block(e3, f, strides=2); f = int(f*2)
        _ = res_block(e4, f, strides=2)

        ## bottleneck
        b0 = conv_block(_, f, strides=1)
        _ = conv_block(b0, f, strides=1)

        ## upsample
        _ = upsamp_concat_block(_, e4)
        _ = res_block(_, f); f = int(f/2)

        _ = upsamp_concat_block(_, e3)
        _ = res_block(_, f); f = int(f/2)

        _ = upsamp_concat_block(_, e2)
        _ = res_block(_, f); f = int(f/2)

        _ = upsamp_concat_block(_, e1)
        _ = res_block(_, f)

        ## classify
        if nclasses==1:
            outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="sigmoid")(_)
        else:
            outputs = tf.keras.layers.Conv2D(nclasses, (1, 1), padding="same", activation="softmax")(_)

        #model creation
        model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
        return model

    #-----------------------------------
    def mean_iou(y_true, y_pred):
        """
        mean_iou(y_true, y_pred)
        This function computes the mean IoU between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

        INPUTS:
            * y_true: true masks, one-hot encoded.
                * Inputs are B*W*H*N tensors, with
                    B = batch size,
                    W = width,
                    H = height,
                    N = number of classes
            * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
                * Inputs are B*W*H*N tensors, with
                    B = batch size,
                    W = width,
                    H = height,
                    N = number of classes
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: None
        OUTPUTS:
            * IoU score [tensor]
        """
        yt0 = y_true[:,:,:,0]
        yp0 = tf.keras.backend.cast(y_pred[:,:,:,0] > 0.5, 'float32')
        inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
        union = tf.math.count_nonzero(tf.add(yt0, yp0))
        iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
        return iou

    #-----------------------------------
    def dice_coef(y_true, y_pred):
        """
        dice_coef(y_true, y_pred)

        This function computes the mean Dice coefficient between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

        INPUTS:
            * y_true: true masks, one-hot encoded.
                * Inputs are B*W*H*N tensors, with
                    B = batch size,
                    W = width,
                    H = height,
                    N = number of classes
            * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
                * Inputs are B*W*H*N tensors, with
                    B = batch size,
                    W = width,
                    H = height,
                    N = number of classes
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: None
        OUTPUTS:
            * Dice score [tensor]
        """
        smooth = 1.
        y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
        y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        """
        dice_coef_loss(y_true, y_pred)

        This function computes the mean Dice loss (1 - Dice coefficient) between `y_true` and `y_pred`: this version is tensorflow (not numpy) and is used by tensorflow training and evaluation functions

        INPUTS:
            * y_true: true masks, one-hot encoded.
                * Inputs are B*W*H*N tensors, with
                    B = batch size,
                    W = width,
                    H = height,
                    N = number of classes
            * y_pred: predicted masks, either softmax outputs, or one-hot encoded.
                * Inputs are B*W*H*N tensors, with
                    B = batch size,
                    W = width,
                    H = height,
                    N = number of classes
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: None
        OUTPUTS:
            * Dice loss [tensor]
        """
        return 1.0 - dice_coef(y_true, y_pred)

    #---------------------------------------------------
    # learning rate function
    def lrfn(epoch):
        """
        lrfn(epoch)
        This function creates a custom piecewise linear-exponential learning rate function for a custom learning rate scheduler. It is linear to a max, then exponentially decays

        * INPUTS: current `epoch` number
        * OPTIONAL INPUTS: None
        * GLOBAL INPUTS:`start_lr`, `min_lr`, `max_lr`, `rampup_epochs`, `sustain_epochs`, `exp_decay`
        * OUTPUTS:  the function lr with all arguments passed

        """
        def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
            if epoch < rampup_epochs:
                lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
            elif epoch < rampup_epochs + sustain_epochs:
                lr = max_lr
            else:
                lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
            return lr
        return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)


    #-----------------------------------
    def seg_labelfile2tensor(f):
        """
        "seg_file2tensor(f)"
        This function reads a jpeg image from file into a cropped and resized tensor,
        for use in prediction with a trained segmentation model
        INPUTS:
            * f [string] file name of jpeg
        OPTIONAL INPUTS: None
        OUTPUTS:
            * image [tensor array]: unstandardized image
        GLOBAL INPUTS: TARGET_SIZE
        """
        bits = tf.io.read_file(f)
        image = tf.image.decode_jpeg(bits)

        w = tf.shape(image)[0]
        h = tf.shape(image)[1]
        tw = TARGET_SIZE[0]
        th = TARGET_SIZE[1]
        resize_crit = (w * th) / (h * tw)
        image = tf.cond(resize_crit < 1,
                      lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                      lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                     )

        nw = tf.shape(image)[0]
        nh = tf.shape(image)[1]
        image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
        # image = tf.cast(image, tf.uint8) #/ 255.0

        return image


    #-----------------------------------
    def seg_file2tensor(f, fir):
        """
        "seg_file2tensor(f)"
        This function reads a jpeg image from file into a cropped and resized tensor,
        for use in prediction with a trained segmentation model
        INPUTS:
            * f [string] file name of jpeg
        OPTIONAL INPUTS: None
        OUTPUTS:
            * image [tensor array]: unstandardized image
        GLOBAL INPUTS: TARGET_SIZE
        """
        bits = tf.io.read_file(f)
        image = tf.image.decode_jpeg(bits)

        bits = tf.io.read_file(fir)
        nir = tf.image.decode_jpeg(bits)

        image = tf.concat([image, nir],-1)[:,:,:4]

        w = tf.shape(image)[0]
        h = tf.shape(image)[1]
        tw = TARGET_SIZE[0]
        th = TARGET_SIZE[1]
        resize_crit = (w * th) / (h * tw)
        image = tf.cond(resize_crit < 1,
                      lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                      lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                     )

        nw = tf.shape(image)[0]
        nh = tf.shape(image)[1]
        image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
        # image = tf.cast(image, tf.uint8) #/ 255.0

        return image

    # #-----------------------------------
    # def gen_dice(y_true, y_pred, eps=1e-6):
    #     """both tensors are [b, h, w, classes] and y_pred is in logit form"""
    #
    #     # [b, h, w, classes]
    #     #pred_tensor = tf.nn.softmax(y_pred)
    #     y_true_shape = tf.shape(y_true)
    #
    #     # [b, h*w, classes]
    #     y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
    #     y_pred = tf.reshape(y_pred, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]]) #pred_tensor
    #
    #     # [b, classes]
    #     # count how many of each class are present in
    #     # each image, if there are zero, then assign
    #     # them a fixed weight of eps
    #     counts = tf.reduce_sum(y_true, axis=1)
    #     weights = 1. / (counts ** 2)
    #     weights = tf.where(tf.math.is_finite(weights), weights, eps)
    #
    #     multed = tf.reduce_sum(y_true * y_pred, axis=1)
    #     summed = tf.reduce_sum(y_true + y_pred, axis=1)
    #
    #     # [b]
    #     numerators = tf.reduce_sum(weights*multed, axis=-1)
    #     denom = tf.reduce_sum(weights*summed, axis=-1)
    #     dices = 1. - 2. * numerators / denom
    #     dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    #     return tf.reduce_mean(dices)
    #
    #
    # def dice_coef(y_true, y_pred):
    #     return 1-gen_dice(y_true, y_pred)


    @tf.autograph.experimental.do_not_convert
    #-----------------------------------
    def read_seg_tfrecord_multiclass(example):
        """
        "read_seg_tfrecord_multiclass(example)"
        This function reads an example from a TFrecord file into a single image and label
        This is the "multiclass" version for imagery, where the classes are mapped as follows:
        INPUTS:
            * TFRecord example object
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: TARGET_SIZE
        OUTPUTS:
            * image [tensor array]
            * class_label [tensor array]
        """
        nclasses=5

        features = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
            "nir_image": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
            "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
        }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        image = tf.image.decode_jpeg(example['image'], channels=3)
        image = tf.cast(image, tf.float32)/ 255.0
        image = tf.reshape(image, [TARGET_SIZE[0],TARGET_SIZE[1], 3])
        #print(image.shape)
        #image = tf.reshape(tf.image.rgb_to_grayscale(image), [TARGET_SIZE,TARGET_SIZE, 1])

        nir = tf.image.decode_jpeg(example['nir_image'], channels=3)
        nir = tf.cast(nir, tf.float32)/ 255.0
        nir = tf.reshape(nir, [TARGET_SIZE[0],TARGET_SIZE[1], 3])
        #print(nir.shape)

        image = tf.concat([image, nir],-1)[:,:,:4]
        #image = tf.cast(np.dstack(image.numpy(), nir.numpy()), tf.float32)
        #print(image.shape)

        label = tf.image.decode_jpeg(example['label'], channels=1)
        label = tf.cast(label, tf.uint8)#/ 255.0
        label = tf.reshape(label, [TARGET_SIZE[0],TARGET_SIZE[1], 1])

        # cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*0)
        # label = tf.where(cond,  tf.ones(tf.shape(label),dtype=tf.uint8)*4, label)
        #print(label.shape)

        label = tf.one_hot(tf.cast(label, tf.uint8), nclasses) # 5 classes (water, surf, wet, dry) + null (0)

        label = tf.squeeze(label)

        image = tf.reshape(image, (image.shape[0], image.shape[1], image.shape[2]))

        #image = tf.image.per_image_standardization(image)
        return image, label


    #-----------------------------------
    def get_batched_dataset(filenames):
        """
        "get_batched_dataset(filenames)"
        This function defines a workflow for the model to read data from
        tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
        and also formats the imagery properly for model training
        INPUTS:
            * filenames [list]
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: BATCH_SIZE, AUTO
        OUTPUTS: tf.data.Dataset object
        """
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = True

        dataset = tf.data.Dataset.list_files(filenames)
        dataset = dataset.with_options(option_no_order)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
        dataset = dataset.map(read_seg_tfrecord_multiclass, num_parallel_calls=AUTO)
        #dataset = dataset.cache() # This dataset fits in RAM
        dataset = dataset.repeat()
        #dataset = dataset.shuffle(2048)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
        dataset = dataset.prefetch(AUTO) #

        return dataset

    ###############################################################
    ### DATA FUNCTIONS
    ###############################################################
    #-----------------------------------
    def get_training_dataset(training_filenames):
        """
        This function will return a batched dataset for model training
        INPUTS: None
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: training_filenames
        OUTPUTS: batched data set object
        """
        return get_batched_dataset(training_filenames)

    def get_validation_dataset(validation_filenames):
        """
        This function will return a batched dataset for model training
        INPUTS: None
        OPTIONAL INPUTS: None
        GLOBAL INPUTS: validation_filenames
        OUTPUTS: batched data set object
        """
        return get_batched_dataset(validation_filenames)

    ###############################################################
    ## VARIABLES
    ###############################################################

    data_path= os.getcwd()+os.sep+"data/sentinel2-tfrecords"

    # sample_data_path = os.getcwd()+os.sep+"data/obx/sample"
    # sample_label_data_path = os.getcwd()+os.sep+"data/obx/sample_labels"

    trainsamples_fig = os.getcwd()+os.sep+'results/s2_sample_4class_trainsamples.png'
    valsamples_fig = os.getcwd()+os.sep+'results/s2_sample_4class_valsamples.png'

    augsamples_fig = os.getcwd()+os.sep+'results/s2_sample_4class_augtrainsamples.png'

    weights = os.getcwd()+os.sep+'results/s2_4class_best_weights_batch_'+str(BATCH_SIZE)+'.h5'

    hist_fig = os.getcwd()+os.sep+'results/s2_trainhist_4class_model_'+str(BATCH_SIZE)+'.png'

    test_samples_fig = os.getcwd()+os.sep+'results/pred/s2_sample_4class.png'

    lr_fig = weights.replace('.h5','_lr.png')

    sample_data_path =  os.getcwd()+os.sep+'sentinel2/images/images'
    sample_label_data_path =  os.getcwd()+os.sep+'sentinel2/labels/labels'


    patience =25
    nclasses=5 # [null,water,surf,wet,dry]
    do_train = True #False # True


    ###############################################################
    ## EXECUTION
    ###############################################################

    #-------------------------------------------------
    filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

    print('.....................................')
    print('Reading files and making datasets ...')

    nb_images = ims_per_shard * len(filenames)
    print(nb_images)

    split = int(len(filenames) * VALIDATION_SPLIT)

    training_filenames = filenames[split:]
    validation_filenames = filenames[:split]

    validation_steps = int(nb_images  // BATCH_SIZE)
    steps_per_epoch = int(nb_images  // BATCH_SIZE)

    # validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
    # steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE

    print(steps_per_epoch)
    print(validation_steps)

    n = len(training_filenames)*ims_per_shard
    print("training files: %i" % (n))
    n = len(validation_filenames)*ims_per_shard
    print("validation files: %i" % (n))

    train_ds = get_training_dataset(training_filenames)
    val_ds = get_validation_dataset(validation_filenames)

    for imgs,lbls in train_ds.take(1):
        print(imgs.shape)
        print(lbls.shape)

    print('.....................................')
    print('Printing examples to file ...')

    plt.figure(figsize=(16,16))
    for imgs,lbls in train_ds.take(1):
      #print(lbls)
      for count,(im,lab) in enumerate(zip(imgs, lbls)):
         plt.subplot(int(BATCH_SIZE/2),2,count+1)
         plt.imshow(im)
         plt.imshow(np.argmax(lab,-1), cmap='bwr', alpha=0.5, vmin=0, vmax=nclasses)
         plt.axis('off')
         #print(np.unique(np.argmax(lab,-1).flatten()))
    # plt.show()
    plt.savefig(trainsamples_fig.replace('.png', '_multiclass.png'), dpi=200, bbox_inches='tight')
    plt.close('all')

    del imgs, lbls

    plt.figure(figsize=(16,16))
    for imgs,lbls in val_ds.take(1):

      #print(lbls)
      for count,(im,lab) in enumerate(zip(imgs, lbls)):
         plt.subplot(int(BATCH_SIZE/2),2,count+1) #int(BATCH_SIZE/2)
         plt.imshow(im)
         plt.imshow(np.argmax(lab,-1), cmap='bwr', alpha=0.5, vmin=0, vmax=nclasses)
         plt.axis('off')
    # plt.show()
    plt.savefig(valsamples_fig.replace('.png', '_multiclass.png'), dpi=200, bbox_inches='tight')
    plt.close('all')
    del imgs, lbls

    rng = [i for i in range(MAX_EPOCHS)]
    y = [lrfn(x) for x in rng]
    plt.plot(rng, [lrfn(x) for x in rng])
    # plt.show()
    plt.savefig(lr_fig, dpi=200, bbox_inches='tight')


    print('.....................................')
    print('Creating and compiling model ...')

    model = res_unet((TARGET_SIZE[0], TARGET_SIZE[1], 4), BATCH_SIZE, nclasses)
    # model.compile(optimizer = 'adam', loss = gen_dice, metrics = [mean_iou, dice_coef])
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [mean_iou, dice_coef])

    # model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = [tf.keras.metrics.MeanIoU(num_classes=nclasses), dice_coef])


    earlystop = EarlyStopping(monitor="val_loss",
                                  mode="min", patience=patience)

    # set checkpoint file
    model_checkpoint = ModelCheckpoint(weights, monitor='val_loss',
                                    verbose=0, save_best_only=True, mode='min',
                                    save_weights_only = True)


    # models are sensitive to specification of learning rate. How do you decide? Answer: you don't. Use a learning rate scheduler

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

    callbacks = [model_checkpoint, earlystop, lr_callback]

    if do_train:
        print('.....................................')
        print('Training model ...')
        history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                              validation_data=val_ds, validation_steps=validation_steps,
                              callbacks=callbacks)

        # Plot training history
        plot_seg_history_iou(history, hist_fig)

        plt.close('all')
        K.clear_session()

    else:
        model.load_weights(weights)


    # ##########################################################
    # ### evaluate
    print('.....................................')
    print('Evaluating model ...')
    # # testing
    # scores = model.evaluate(val_ds, steps=validation_steps)
    #
    # print('loss={loss:0.4f}, Mean IoU={mean_iou:0.4f}'.format(loss=scores[0], mean_iou=scores[1]))
    #
    # #75%
    #
    # # ##########################################################

    median_filter_value = 2
    doplot=True #False
    IOUc = []

    counter = 0
    for i,l in val_ds.take(10):

        for img,lbl in zip(i,l):
            #print(i.shape)
            #image = seg_file2tensor(f)/255
            est_label = model.predict(tf.expand_dims(img, 0) , batch_size=1).squeeze() #tf.expand_dims(i, 0) , batch_size=1).squeeze()
            est_label = tf.argmax(est_label, axis=-1)

            #est_label,_ = crf_refine(est_label.numpy(), im.numpy().astype(np.uint8), theta_col=100, theta_spat=3)
            #for e,im in zip(est_label,i):
            est_label = np.round(median(est_label, disk(median_filter_value))).astype(np.uint8)
            est_label[est_label==0] = 1
            #lbls.append(est_label)

            lbl = tf.argmax(lbl, axis=-1).numpy()
            lbl[lbl==0]=1
            lbl = np.round(median(lbl, disk(median_filter_value))).astype(np.uint8)

            if doplot:
                plt.subplot(221)
                plt.imshow(img[:,:,:3])
                plt.imshow(lbl, alpha=0.4, cmap=plt.cm.bwr, vmin=1, vmax=nclasses)
                plt.axis('off')

                plt.subplot(222)
                plt.imshow(img)
                plt.imshow(est_label, alpha=0.4, cmap=plt.cm.bwr, vmin=1, vmax=nclasses)
                plt.axis('off')
                iouscore = iou(lbl, est_label, nclasses)
                plt.title('iou = '+str(iouscore)[:5])
                IOUc.append(iouscore)

                plt.savefig(test_samples_fig.replace('.png', '_val_'+str(counter)+'.png'),
                        dpi=200, bbox_inches='tight')
                plt.close('all')
            counter += 1

    print('Mean IoU={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
    #55
    #55
    #55
    #55


    ### predict
    print('.....................................')
    print('Using model for prediction on jpeg images ...')


    sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))
    print('Number of samples: %i' % (len(sample_filenames)))

    sample_label_data_path = sample_data_path.replace('images', 'labels')
    sample_label_filenames = sorted(tf.io.gfile.glob(sample_label_data_path+os.sep+'*.jpg'))

    IOUc = []
    median_filter_value = 3

    for counter,f in enumerate(sample_filenames):
        image = seg_file2tensor(f, f.replace('aug_images', 'aug_nir') )/255
        #print(image.shape)

        lbl = seg_labelfile2tensor(sample_label_filenames[counter])
        lbl = np.round(lbl[:,:,0])

        est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
        est_label = tf.argmax(est_label, axis=-1)

        #est_label,_ = crf_refine(est_label.numpy(), im.numpy().astype(np.uint8), theta_col=100, theta_spat=3)
        est_label = np.round(median(est_label, disk(median_filter_value))).astype(np.uint8)
        est_label[est_label==0] = 1

        name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]

        plt.figure() #figsize=(16,16))
        plt.imshow(image[:,:,:3])
        plt.imshow(est_label, alpha=0.4, cmap=plt.cm.bwr, vmin=1, vmax=nclasses-1)
        plt.axis('off')
        iouscore = iou(lbl, est_label, nclasses)
        plt.title('iou = '+str(iouscore)[:5])
        IOUc.append(iouscore)
        # plt.show()
        plt.savefig(test_samples_fig.replace('.png', '_sample_'+str(counter)+'.png'),
                dpi=200, bbox_inches='tight')
        plt.close('all')


    print('Mean IoU={mean_iou:0.3f}'.format(mean_iou=np.mean(IOUc)))
    #47
    #48
    #48
    #48
