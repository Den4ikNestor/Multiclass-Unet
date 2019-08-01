# WISH YOU TO BE SATISFIED OF U-NET MODEL TRAINING PROCESS
import numpy as np
import os
import random
import re
from PIL import Image
import sys
from pylab import *
import cv2
from datetime import datetime
import argparse
import tensorflow as tf
tf.enable_eager_execution()

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute, add
from tensorflow.compat.v1.layers import conv2d_transpose
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.client import device_lib

print('TENSORFLOW VERSION:  ' + tf.__version__ + '\n')
print('LOCAL DEVICES\n' + device_lib.list_local_devices())

x = tf.random.uniform([3, 3])
print('\n\nIs tensorflow using \'Eager execution\' function: {}'.format(tf.executing_eagerly()))
print("Is there a GPU available:    {}".format(tf.test.is_gpu_available()))
print("Is the Tensor on GPU #0:    {}".format(x.device.endswith('GPU:0')))
print("Device name:    {}\n\n".format((x.device)))

class Unet():
    '''
    The class Unet for implementing full-connected U-net CNN
    The model can predict more then just 1 class, just set 'n_classes' parameter
    '''
    def __init__(self, n_classes=2, im_width=256, im_height=256, batch_size=5, num_epochs=100, steps_per_epoch=30):
        '''
        Input:
            * n_classes - the number of predicted classes
            * im_width - the width of input image
            * im_height - the height of input image
            * batch_size - tensor parameter
            * num_epochs - the number of epochs. Parameter in training method
            * steps_per_epoch - Parameter in training method. How 'deep' the model is training per one epoch
        '''
        self.n_classes = n_classes
        self.im_width = im_width
        self.im_height = im_height
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch    

    
    def _read_to_tensor(fname, output_height=256, output_width=256):
        '''Function to read images from given image file path, and provide resized images as tensors
            Inputs: 
                fname - image file path
                output_height - required output image height
                output_width - required output image width
            Output: Processed image tensors
        '''
        # Read the image and decode a JPEG-encoded image to a uint8 tensor
        img_strings = tf.io.read_file(fname)
        imgs_decoded = tf.image.decode_jpeg(img_strings)
        
        # Resize the image
        output = tf.image.resize(imgs_decoded, [output_height, output_width])
        
        return output


    def read_images(img_dir):
        '''
        Function to get all image directories, read images and masks in separate tensors
            Inputs: 
                img_dir - file directory
            Outputs 
                frame_tensors, masks_tensors, frame files list, mask files list
        '''
        # Get the file names list from provided directory
        file_list = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        
        # Separate frame and mask files lists, exclude unnecessary files
        frames_list = [file for file in file_list if ('_mask' not in file) and ('txt' not in file)]
        masks_list = [file for file in file_list if ('_mask' in file) and ('txt' not in file)]
        
        print('{} frame files found in the provided directory.'.format(len(frames_list)))
        print('{} mask files found in the provided directory.'.format(len(masks_list)))
        
        # Create full file paths to images
        frames_paths = [os.path.join(img_dir, fname) for fname in frames_list]
        masks_paths = [os.path.join(img_dir, fname) for fname in masks_list]
        
        # Create dataset of tensors
        frame_data = tf.data.Dataset.from_tensor_slices(frames_paths)
        masks_data = tf.data.Dataset.from_tensor_slices(masks_paths)
        
        # Read images into the tensor dataset
        frame_tensors = frame_data.map(_read_to_tensor)
        masks_tensors = masks_data.map(_read_to_tensor)
        print('The converting images and masks to tensors were complited')
        
        return frame_tensors, masks_tensors, frames_list, masks_list


    def create_folder(data_path):
        '''
        Create folders for keeping training and validation images and masks
        '''
        folders = ['train_frames', 'train_masks', 'val_frames', 'val_masks', 'prep_train_frames', 'prep_train_masks']

        for folder in folders:
            try:
                os.makedirs(data_path + folder)
            except Exception as e: print(e)


    def generate_image_folder_structure(data_path, frames, masks, frames_list, masks_list):
        '''
        Function to save images in the appropriate folder directories 
        Inputs:
            * data_path - the parent of folders 'train' and 'val' for images and masks
            * frames - frame tensor dataset
            * masks - mask tensor dataset
            * frames_list - frame file paths
            * masks_list - mask file paths
        '''
        #Create iterators for frames and masks
        frame_batches = tf.compat.v1.data.make_one_shot_iterator(frames)
        mask_batches = tf.compat.v1.data.make_one_shot_iterator(masks)
        
        # Save masks and images into 'data_path/train_(masks, frames)'
        dir_name='train'
        for file in zip(frames_list[:-round(0.2*len(frames_list))],masks_list[:-round(0.2*len(masks_list))]):
            #Convert tensors to numpy arrays
            frame = frame_batches.next().numpy().astype(np.uint8)
            mask = mask_batches.next().numpy().astype(np.uint8)
            
            #Convert numpy arrays to images
            frame = Image.fromarray(frame)
            mask = Image.fromarray(mask)
            
            #Save frames and masks to correct directories
            frame.save(data_path + '{}_frames/'.format(dir_name) + file[0])
            mask.save(data_path + '{}_masks/'.format(dir_name) + file[1])
        
        # Save masks and images into 'data_path/val_(masks, frames)'
        dir_name='val'
        for file in zip(frames_list[-round(0.2*len(frames_list)):],masks_list[-round(0.2*len(masks_list)):]):
            #Convert tensors to numpy arrays
            frame = frame_batches.next().numpy().astype(np.uint8)
            mask = mask_batches.next().numpy().astype(np.uint8)
            
            #Convert numpy arrays to images
            frame = Image.fromarray(frame)
            mask = Image.fromarray(mask)
            
            #Save frames and masks to correct directories
            frame.save(data_path + '{}_frames/'.format(dir_name) + file[0])
            mask.save(data_path + '{}_masks/{}'.format(dir_name) + file[1])
        
        print("The images and masks were saved in appropriate folder for training and validation in the path \'{}\'".format(data_path))


    def make_another_classes_black(class_color_masks, another_color_masks = None):
        mask_path_train = './CamSeq01/train_masks/train/'
        mask_path_val = './CamSeq01/val_masks/val/'
        save_train_dir = './CamSeq01/train_masks/prep/'
        save_val_dir = './CamSeq01/val_masks/prep/'
        
        list_of_black_masks_train = os.listdir(mask_path_train)
        list_of_black_masks_val = os.listdir(mask_path_val)
        
        for image_name in list_of_black_masks_train:
            image = Image.open(mask_path_train + image_name)
            image_mask = np.array(image)
            for col in class_color_masks:
                image_mask[np.where((image_mask != col).all(axis = 2))] = [255,255,255]
            # for col in another_color_masks:
                # image_mask[np.where((image_mask == col).all(axis = 2))] = 
                # [255,255,255]
            
            image = Image.fromarray(image_mask.astype('uint8'), 'RGB')
            image.save(save_train_dir+image_name, 'PNG')
        
        for image_name in list_of_black_masks_val:
            image = Image.open(mask_path_val + image_name)
            image_mask = np.array(image)
            for col in class_color_masks:
                image_mask[np.where((image_mask != col).all(axis = 2))] = [255,255,255]
            # for col in another_color_masks:
                # image_mask[np.where((image_mask == col).all(axis = 2))] = 
                # [255,255,255]
            image = Image.fromarray(image_mask.astype('uint8'), 'RGB')
            image.save(save_val_dir + image_name, 'PNG')


    def preprocess_images(data_path, train_val_ration=0.8, class_color_codes = []):
        '''
        * Find masks and images
        * Convert them to tensors
        * Create folders for training and validation in 'data_path'
        * Preprocess masks painting extra classes to white color
        Inputs:
            * data_path - the way to data path
            * train_val_ratio - the ration between train and validation datasets
            * class_color_codes - the list of class codes in RGB tuple format which the algorith will use for training  e.g. [(0,192,255), (...), ..]
        '''
        
        
        
    def get_unet_model(data_type = 'image', n_filters = 32, n_classes = 1, activation_func = 'sigmoid', dilation_rate = 1, input_size = (256,256,1)):
        '''
        Validation Image data generator
        Inputs: 
            * data_type - the type of input data {'image' or 'tensor'}
            * n_filters - base convolution filters [recommended to use the powers of 2: (.., 32, 64, 128, ..)]
            * n_classes - the final prediction of N classes (int)
            * activation_func - the final layer activation function {'sigmoid', 'softmax', ...}
            * dilation_rate - convolution dilation rate [number of blank spaces (dilation_rate - 1) between each pixel in kernel]
        Output: Unet keras Model
        '''
        # Check the correctness of 'data_type' and 'input_size'
        data_format = None
        inputs = None
        if data_type == 'image': 
            if __debug__:
                if not len(input_size) == 3: 
                    print('Incorrect input_size for image data. It should have 3 dimensions')
                    raise AssertionError
                inputs = Input(input_size)
        elif data_type == 'tensor': 
            if __debug__:
                if not len(input_size) == 4: 
                    print('Incorrect input_size for tensor data. It should have 4 dimansions')
                    raise AssertionError
                inputs = Input(batch_shape = input_size)
                data_format = 'channels_last'

        # filters - the dimensionality of output space
        # kernel_size - the tuple (height, width) or just int number of convolution window
        # activation - nonlinear function is needed for the neural network won't be just linear regression
        # padding - ['same' - the dimensions after convolution process are the same as input value; 
        #            'valid' - the dimensions of output can be changes in convolution process]
        # dilation_rate - convolution dilation rate

        # pool_size - the size of pooling window
        # data_format - the ordering of the dimensions in the inputs (for image type the value should be 'None')

        conv1 = Conv2D(filters = n_filters * 1, kernel_size = (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv1)

        conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv2)

        conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv3)

        conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv4)

        conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(n_filters * 16, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv5)
        conv5 = BatchNormalization()(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
        conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(n_filters * 8, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
        conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(n_filters * 4, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
        conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(n_filters * 2, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
        conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(n_filters * 1, (3, 3), activation='relu', padding = 'same', dilation_rate = dilation_rate)(conv9)
        conv9 = BatchNormalization()(conv9)

        # softmax - multiclass classification activation function
        # sigmoid - the prediction of probality
        conv10 = Conv2D(n_classes, (1, 1), activation=activation_func, padding = 'same', dilation_rate = dilation_rate)(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model


    def tversky_loss(y_true, y_pred):
        '''
        Training metric
        '''
        alpha = 0.5
        beta  = 0.5

        ones = K.ones(K.shape(y_true))
        p0 = y_pred      # proba that voxels are class i
        p1 = ones-y_pred # proba that voxels are not class i
        g0 = y_true
        g1 = ones-y_true

        num = K.sum(p0*g0, (0,1,2,3))
        den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))

        T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

        Ncl = K.cast(K.shape(y_true)[-1], 'float32')
        return Ncl-T

   
    def dice_coef(y_true, y_pred):
        '''
        Training metric
        '''
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


    def dice_coef_loss(y_true, y_pred):
        return 1.-dice_coef(y_true, y_pred)
    

    def prep_model(self):
        config = tf.ConfigProto()

        #config.gpu_options.per_process_gpu_memory_fraction = 0.3#0.8  # or any valid options.
        #config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        activation_function = ''
        if self.n_classes >= 2: activation_function = 'softmax'
        else: activation_function = 'sigmoid'
        
        model = self.get_unet_model(data_type = 'tensor', n_filters = 32, self.n_classes, activation_func=activation_function, dilation_rate = 1, 
        input_size = (self.batch_size, self.im_height, self.im_width, 3))
        
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[tversky_loss,dice_coef,'acc'])  ## binary_crossentropy
        
        return model


    def prepare_nn(self, weights = ''):
        model = self.prep_model()
        try:
            model.load_weights(weights)
        except:
            print('Weights are not found. Let\'s train the model')
        return model


    def define_model_callbacks():
        tb = TensorBoard(log_dir='logs', write_graph=True)
        mc = ModelCheckpoint(mode='max', filepath='unet_checkpoint.h5', monitor='acc', save_best_only='True', save_weights_only='True', verbose=1)
        es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=1)
        callbacks = [tb, mc, es]
        return callbacks


# code2id = {v:k for k,v in enumerate(label_code)}
# id2code = {k:v for k,v in enumerate(label_code)}

# name2id = {v:k for k,v in enumerate(label_names)}
# id2name = {k:v for k,v in enumerate(label_names)}


    def rgb_to_onehot(rgb_image, colormap = id2code_4):
        '''Function to one hot encode RGB mask labels
            Inputs: 
                rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
                colormap - dictionary of color to label id
            Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
        '''
        num_classes = len(colormap)
        shape = rgb_image.shape[:2]+(num_classes,)
        encoded_image = np.zeros( shape, dtype=np.int8 )
        for i, cls in enumerate(colormap):
            encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
        return encoded_image


    def onehot_to_rgb(onehot, colormap = id2code_4):
        '''Function to decode encoded mask labels
            Inputs: 
                onehot - one hot encoded image matrix (height x width x num_classes)
                colormap - dictionary of color to label id
            Output: Decoded RGB image (height x width x 3) 
        '''
        single_layer = np.argmax(onehot, axis=-1)
        output = np.zeros( onehot.shape[:2]+(3,) )
        for k in colormap.keys():
            output[single_layer==k] = colormap[k]
        return np.uint8(output)


                            # Normalizing only frame images, since masks contain label info
                            data_gen_args = dict(rescale=1./255)
                            mask_gen_args = dict()

                            train_frames_datagen = ImageDataGenerator(**data_gen_args)
                            train_masks_datagen = ImageDataGenerator(**mask_gen_args)
                            val_frames_datagen = ImageDataGenerator(**data_gen_args)
                            val_masks_datagen = ImageDataGenerator(**mask_gen_args)

                            # Seed defined for aligning images and their masks
                            seed = 1


    def TrainAugmentGenerator(path_data = '', batch_size = 5):
        '''Train Image data generator
            Inputs: 
                seed - seed provided to the flow_from_directory function to ensure aligned data flow
                batch_size - number of images to import at a time
            Output: Decoded RGB image (height x width x 3) 
        '''
        seed = 1
        train_image_generator = train_frames_datagen.flow_from_directory(
        path_data + 'train_frames/',
        batch_size = batch_size, seed = seed)

        train_mask_generator = train_masks_datagen.flow_from_directory(
        path_data + 'train_masks/',
        batch_size = batch_size, seed = seed)

        while True:
            X1i = train_image_generator.next()
            X2i = train_mask_generator.next()
            
            #One hot encoding RGB images
            mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code_4) for x in range(X2i[0].shape[0])]
            
            yield X1i[0], np.asarray(mask_encoded)


    def ValAugmentGenerator(path_data = '', batch_size = 5):
        '''Validation Image data generator
            Inputs: 
                seed - seed provided to the flow_from_directory function to ensure aligned data flow
                batch_size - number of images to import at a time
            Output: Decoded RGB image (height x width x 3) 
        '''
        seed = 1
        val_image_generator = val_frames_datagen.flow_from_directory(
        path_data + 'val_frames/',
        batch_size = batch_size, seed = seed)

        val_mask_generator = val_masks_datagen.flow_from_directory(
        path_data + 'val_masks/',
        batch_size = batch_size, seed = seed)

        while True:
            X1i = val_image_generator.next()
            X2i = val_mask_generator.next()
            
            #One hot encoding RGB images
            mask_encoded = [rgb_to_onehot(X2i[0][x,:,:,:], id2code_4) for x in range(X2i[0].shape[0])]
            
            yield X1i[0], np.asarray(mask_encoded)


    def train_nn(self):
        '''
        Train model on data
        '''
        model = self.prepare_nn()
        callbacks = define_model_callbacks()
        validation_steps = (float((round(0.1*len(||||FRAMES_LIST|||||)))) / float(self.batch_size))
        
        tstart = datetime.now()
        
        result = model.fit_generator(
        TrainAugmentGenerator(batch_size = self.batch_size), 
        steps_per_epoch=self.steps_per_epoch,
        validation_data = ValAugmentGenerator(batch_size = self.batch_size), 
        validation_steps = validation_steps, 
        epochs=self.num_epochs, 
        callbacks=callbacks)
        
        model.save_weights("unet_model.h5", overwrite=True)
        
        tprep = datetime.now()
        print('{}s for fitting the NN'.format((tprep - tstart).total_seconds()))
        #141.55697s for fitting the NN


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Train Unet.')
    #parser.add_argument('images', required=False, type=str, default='', help='folder with images')
    #args = parser.parse_args()

    #print(args)
    path_data = './data/'

    unet = Unet(n_classes=5, im_width=256, im_height=256, batch_size=5, num_epochs=100, steps_per_epoch=30)
    
    X, y = nn.get_data(path_data, train=True)
    print(X.shape)

    nn.train_nn(train_test_split(X, y, test_size=0.2), 'model-tgs-salt-2classes.h5')

    #t = datetime.now()