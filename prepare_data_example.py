## the following assumes that you have two lists with the file paths, one list of the file paths for the images (img_paths), and one for the masks (mask_paths)

# create arrays from files
import math
from skimage import io as skio
from skimage import util as skutil
import numpy as np

smpl_imgs = []
for i in range(len(img_paths)):
    smpl_imgs.append(skio.imread(img_paths[i])/10000)
smpl_masks = []
for i in range(len(mask_paths)):
    m = skio.imread(mask_paths[i])
    m = np.expand_dims(m, axis=2)
    smpl_masks.append(m)

# you should now have one list of arrays for images and one for masks



## functions for preprocessing and simple augmentation :

# define function for preprocessing, namely converting and resizing (depending on the tile size you need)
import tensorflow as tf

def preprocess_arrays(img_array, mask_array):
   # convert to float and resize
    img = tf.image.convert_image_dtype(img_array,dtype=tf.float32)
    img = tf.image.resize(img, [512, 512])

    mask = tf.image.resize(mask_array, [512, 512])
    

    return img, mask


# define function for augmentation (a very simple one, simply flipping the images)
def dataset_augmentation(dataset):
    #augmentation 1
    augmentation = dataset.map(lambda x, y:(tf.image.flip_left_right(x),tf.image.flip_left_right(y)))
    dataset_augmented = tf.data.Dataset.concatenate(dataset,augmentation)
    #augmentation 2
    augmentation = dataset.map(lambda x, y:(tf.image.flip_up_down(x),tf.image.flip_up_down(y)))
    dataset_augmented = tf.data.Dataset.concatenate(dataset_augmented,augmentation)
    #augmentation 3
    augmentation = dataset.map(lambda x, y:(tf.image.flip_left_right(x),tf.image.flip_left_right(y)))
    augmentation = augmentation.map(lambda x, y:(tf.image.flip_up_down(x),tf.image.flip_up_down(y)))
    dataset_augmented = tf.data.Dataset.concatenate(dataset_augmented,augmentation)
    return dataset_augmented




## prepare dataset using the functions above

BATCHSIZE = 5

#create dataset from sample files
smpl_dataset = tf.data.Dataset.from_tensor_slices((smpl_imgs,smpl_masks))   
smpl_dataset = smpl_dataset.map(preprocess_arrays) 
smpl_dataset = smpl_dataset.shuffle(BATCHSIZE*128, reshuffle_each_iteration=False)

#create training dataset by taking first 80% of smpl_dataset (which after shuffeling above means random selection)
#perform augmentation o of the training_dataset (i.e. augment only the first 80% of the smpl_dataset)
training_dataset = dataset_augmentation(smpl_dataset.take(math.floor(len(smpl_imgs)*0.8)))
training_dataset = training_dataset.shuffle(BATCHSIZE*128, reshuffle_each_iteration=True)
training_dataset = training_dataset.batch(BATCHSIZE)

#create validation dataset by taking remaining 20% of shuffeled smpl_dataset 
validation_dataset = smpl_dataset.skip(math.floor(len(smpl_imgs)*0.8))
validation_dataset = validation_dataset.batch(BATCHSIZE)



#you should now have a training dataset and a validation dataset that you can inout to function fit() of model

#train (assuming you have a compiled model, here simply named "model" )
history = model.fit(training_dataset,validation_data=validation_dataset,epochs=10)