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

# from imageio import imwrite
from skimage.io import imsave

from glob import glob
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels
from skimage.filters.rank import median
from skimage.morphology import disk

#-----------------------------------
def get_seg_dataset_for_tfrecords(imdir, shared_size):
    """
    "get_seg_dataset_for_tfrecords"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This is the version for OBX data, which differs in use of both
    resize_and_crop_seg_image_obx and resize_and_crop_seg_image_obx
    for image pre-processing
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    dataset = tf.data.Dataset.list_files(imdir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
    dataset = dataset.map(read_seg_image_and_label)
    dataset = dataset.map(resize_and_crop_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(shared_size)
    return dataset


#-----------------------------------
def recompress_seg_image(image, nir_image, label):
    """
    "recompress_seg_image"
    This function takes an image and label encoded as a byte string
    and recodes as an 8-bit jpeg
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

    nir_image = tf.cast(nir_image, tf.uint8)
    nir_image = tf.image.encode_jpeg(nir_image, optimize_size=True, chroma_downsampling=False)

    label = tf.cast(label, tf.uint8)
    label = tf.image.encode_jpeg(label, optimize_size=True, chroma_downsampling=False)
    return image, nir_image, label


#-----------------------------------
def read_seg_image_and_label(img_path):
    """
    "read_seg_image_and_label_obx(img_path)"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This works by parsing out the label image filename from its image pair
    Thre are different rules for non-augmented versus augmented imagery
    INPUTS:
        * img_path [tensor string]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [bytestring]
        * label [bytestring]
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
    lab_path = tf.strings.regex_replace(img_path, "images", "labels")
    #lab_path = tf.strings.regex_replace(lab_path, ".jpg", "_label.jpg")
    bits = tf.io.read_file(lab_path)
    label = tf.image.decode_jpeg(bits)

    nir_path = tf.strings.regex_replace(img_path, "images", "nir")
    bits = tf.io.read_file(nir_path)
    nir = tf.image.decode_jpeg(bits)

    return image, nir,label


#-----------------------------------
def resize_and_crop_seg_image(image, nir, label):
    """
    "resize_and_crop_seg_image_obx"
    This function crops to square and resizes an image and label
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
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

    label = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(label, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(label, [w*th/h, h*th/h])  # if false
                 )
    label = tf.image.crop_to_bounding_box(label, (nw - tw) // 2, (nh - th) // 2, tw, th)


    nir = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(nir, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(nir, [w*th/h, h*th/h])  # if false
                 )
    nir = tf.image.crop_to_bounding_box(nir, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return image, nir, label


#-----------------------------------
def write_seg_records(dataset, tfrecord_dir):
    """
    "write_seg_records(dataset, tfrecord_dir)"
    This function writes a tf.data.Dataset object to TFRecord shards
    INPUTS:
        * dataset [tf.data.Dataset]
        * tfrecord_dir [string] : path to directory where files will be written
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (files written to disk)
    """
    for shard, (image, nir, label) in enumerate(dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+"sentinel2-" + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_seg_tfrecord(image.numpy()[i], nir.numpy()[i], label.numpy()[i])
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))


#-----------------------------------
def _bytestring_feature(list_of_bytestrings):
    """
    "_bytestring_feature"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_bytestrings
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

#-----------------------------------
def to_seg_tfrecord(img_bytes, nir_bytes, label_bytes):
    """
    "to_seg_tfrecord"
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes
        * label_bytes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "nir_image": _bytestring_feature([nir_bytes]), # one image in the list
      "label": _bytestring_feature([label_bytes]), # one label image in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))



###############################################################
### PLOTTING FUNCTIONS
###############################################################

# #-----------------------------------
# def crf_refine(label, img, theta_col=100, theta_spat=3):
#     """
#     "crf_refine(label, img)"
#     This function refines a label image based on an input label image and the associated image
#     Uses a conditional random field algorithm using spatial and image features
#     INPUTS:
#         * label [ndarray]: label image 2D matrix of integers
#         * image [ndarray]: image 3D matrix of integers
#     OPTIONAL INPUTS: None
#     GLOBAL INPUTS: None
#     OUTPUTS: label [ndarray]: label image 2D matrix of integers
#     """
#
#     nclasses = 5
#     H = label.shape[0]
#     W = label.shape[1]
#     U = unary_from_labels(1+label,nclasses,gt_prob=0.51)
#     d = dcrf.DenseCRF2D(H, W, nclasses)
#     d.setUnaryEnergy(U)
#
#     # to add the color-independent term, where features are the locations only:
#     d.addPairwiseGaussian(sxy=(theta_spat, theta_spat),
#                  compat=3,
#                  kernel=dcrf.DIAG_KERNEL,
#                  normalization=dcrf.NORMALIZE_SYMMETRIC)
#     feats = create_pairwise_bilateral(
#                           sdims=(theta_col, theta_col),
#                           schan=(2,2,2),
#                           img=img,
#                           chdim=2)
#
#     d.addPairwiseEnergy(feats, compat=120,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
#     Q = d.inference(20)
#     kl1 = d.klDivergence(Q)
#     return np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8), kl1


###############################################################

###############################################################
## VARIABLES
###############################################################

imdir = os.getcwd()+os.sep+'sentinel2/images'
nir_imdir = os.getcwd()+os.sep+'sentinel2/nir'
lab_path = os.getcwd()+os.sep+'sentinel2/labels'

tfrecord_dir = os.getcwd()+os.sep+'data/sentinel2-tfrecords'

median_filter_value = 3

nclasses=5
#=======================


n_im = len(glob(imdir+os.sep+'images/*.jpg'))
print(n_im)

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=0,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     fill_mode='nearest',
                     zoom_range=0)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
nir_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

i = 0
for k in range(10):
    img_generator = image_datagen.flow_from_directory(
            imdir,
            target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
            batch_size=n_im,
            class_mode=None, seed=SEED, shuffle=True)

    nir_generator = nir_datagen.flow_from_directory(
            nir_imdir,
            target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
            batch_size=n_im,
            class_mode=None, seed=SEED, shuffle=True)

    #the seed must be the same as for the training set to get the same images
    mask_generator = mask_datagen.flow_from_directory(
            lab_path,
            target_size=(TARGET_SIZE[0], TARGET_SIZE[1]),
            batch_size=n_im,
            class_mode=None, seed=SEED, shuffle=True)

    #The following merges the two generators (and their flows) together:
    train_generator = (pair for pair in zip(img_generator, nir_generator, mask_generator))

    #grab a batch of images and label images
    x, ii, y = next(train_generator)

    # wrute them to file and increment the counter
    for im,nim,lab in zip(x,ii,y):
        l = np.round(lab[:,:,0]).astype(np.uint8)
        nim = np.round(nim).astype(np.uint8)
        print(nim.shape)
        try:
            #l,_ = crf_refine(l, im.astype(np.uint8), theta_col=100, theta_spat=3)
            l = np.round(median(l, disk(median_filter_value))).astype(np.uint8)
            l[l==0]=1 #null is water
            l[l>4]=4
            print(np.unique(l.flatten()))

            plt.imshow(im.astype(np.uint8))
            plt.imshow(l, cmap='bwr', alpha=0.3, vmin=0, vmax=nclasses)
            plt.axis('off')
            # plt.show()
            plt.savefig('example'+str(i)+'.png', dpi=200, bbox_inches='tight')
            plt.close('all')

            imsave(nir_imdir+os.sep+'aug_nir/augimage_000'+str(i)+'.jpg', nim.astype(np.uint8))
            imsave(imdir+os.sep+'aug_images/augimage_000'+str(i)+'.jpg', im.astype(np.uint8))
            imsave(lab_path+os.sep+'aug_labels/augimage_000'+str(i)+'.jpg', l)
            i += 1
        except:
            pass

    #save memory
    del x, y, im, nim, lab
    #get a new batch

images = sorted(tf.io.gfile.glob(imdir+os.sep+'aug_images'+os.sep+'*.jpg'))
# images = sorted(tf.io.gfile.glob(imdir+os.sep+'images'+os.sep+'*.jpg'))

nb_images=len(images)
print(nb_images)

SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

dataset = get_seg_dataset_for_tfrecords(imdir+os.sep+'aug_images', shared_size) #lab_path+os.sep+'aug_labels',
# dataset = get_seg_dataset_for_tfrecords(imdir+os.sep+'aug_images', lab_path+os.sep+'aug_labels', shared_size)

BATCH_SIZE = 16

counter = 0
# view a batch
for imgs,nirs,lbls in dataset.take(1):
  imgs = imgs[:BATCH_SIZE]
  lbls = lbls[:BATCH_SIZE]
  nirs = nirs[:BATCH_SIZE]

  for count,(im,nim,lab) in enumerate(zip(imgs,nirs,lbls)):
     #plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(tf.image.decode_jpeg(im, channels=3))
     plt.imshow(tf.image.decode_jpeg(lab, channels=1), alpha=0.3, cmap='bwr',vmin=0, vmax=4)
     plt.axis('off')
     #plt.show()
     plt.savefig('ex'+str(counter)+'.png', dpi=200, bbox_inches='tight')
     counter +=1
     plt.close('all')

     plt.imshow(tf.image.decode_jpeg(nim, channels=3))
     plt.imshow(tf.image.decode_jpeg(lab, channels=1), alpha=0.3, cmap='bwr',vmin=0, vmax=4)
     plt.axis('off')
     #plt.show()
     plt.savefig('ex-nir'+str(counter)+'.png', dpi=200, bbox_inches='tight')
     counter +=1
     plt.close('all')

write_seg_records(dataset, tfrecord_dir)

#
