from datetime import datetime
from tqdm.autonotebook import tqdm
from joblib import Parallel, delayed
from piscat.Visualization.display import Display
from piscat.InputOutput.reading_videos import DirectoryType, video_reader
from piscat.InputOutput.cpu_configurations import CPUConfigurations
from piscat.InputOutput.gpu_configurations import GPUConfigurations
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, LeakyReLU, UpSampling2D, MaxPooling2D, ZeroPadding2D, Cropping2D, Concatenate, Reshape, GlobalAveragePooling2D, BatchNormalization, Add, Subtract, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras import Input

import tensorflow as tf
import tensorflow.keras.backend as K

use_bias = False
stateful = False


class FastDVDNet:

    def __init__(self, video_original, num_frames_selected=5):
        '''
        Keras implementation of FastDVDNet for grayscale video.

        Parameters
        ----------
        video_original: NDArray
             The video is 3D-numpy (number of frames, width, height).

        num_frames_selected: int (odd)
            The number of selected frames in each batch. This is an odd number for defining numbers as forwards and backward of the target frame.

        References
        ----------
        [1] Tassano, Matias, Julie Delon, and Thomas Veit. "Fastdvdnet: Towards real-time deep video denoising without
        flow estimation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
        '''

        self.video_original = video_original
        self.num_frames_selected = num_frames_selected

        self.cpu = CPUConfigurations()
        self.gpu = GPUConfigurations()

        self.patch_gen = None
        self.patch_original = None
        self.patch_DL_background = None
        self.patch_DL_particles = None
        self.patch_test = None

    def data_handling(self, stride=1):
        '''
        data handling
        Prepare data in the following shape for feeding to FastDVDNet: (number of batch, image size x, image size y, number of selected frames)

        Parameters
        ----------
        stride: int
            stride between forwards and backwards frames

        Returns
        -------
        batch_original_array: NDArray
            The video is 4D-numpy (number of batch, image size x, image size y, number of selected frames).

        video_original: NDArray
            The cropped input video for feeding to u-net.
        '''

        F, H, W = self.video_original.shape
        min_dim = np.min((H, W))
        center_image_x = int(round(H/2))
        center_image_y = int(round(W/2))
        w_x = w_y = int(0.5 * 8 *  int(min_dim/8))
        self.video_original = self.video_original[:, center_image_x-w_x:center_image_x+w_x, center_image_y-w_y:center_image_y+w_y]

        batch_original = [self.video_original[i_:(i_ + stride * self.num_frames_selected):stride, :, :] for i_ in
                          range(0, self.video_original.shape[0] - (stride * self.num_frames_selected) + 1)]

        batch_original_array = np.empty(
            (len(batch_original), batch_original[0].shape[1], batch_original[0].shape[2], batch_original[0].shape[0]))

        for i_, v_orig in enumerate(batch_original):
            tmp_orignal = v_orig.transpose(1, 2, 0)

            batch_original_array[i_, :, :, :] = tmp_orignal

        return batch_original_array, self.video_original

    def pixel_shuffle(self, scale):
        return lambda x: tf.nn.depth_to_space(x, scale)

    def u_net(self, in_put):
        d = in_put
        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = tf.keras.layers.DepthwiseConv2D(depth_multiplier=30, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1),
                            padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=32, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        skip_0 = d

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=64, use_bias=use_bias, kernel_size=(3, 3), strides=(2, 2), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=64, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=64, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        skip_1 = d

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=128, use_bias=use_bias, kernel_size=(3, 3), strides=(2, 2), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=128, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=128, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=128, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=128, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=256, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = Lambda(self.pixel_shuffle(scale=2))(d)
        d = tf.keras.layers.Add()([d, skip_1])

        ###### Up

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=64, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=64, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=128, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = Lambda(self.pixel_shuffle(scale=2))(d)
        d = tf.keras.layers.Add()([d, skip_0])

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=32, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)
        d = BatchNormalization()(d)
        d = tf.keras.layers.Activation('relu')(d)

        d = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(d)
        d = Conv2D(filters=1, use_bias=use_bias, kernel_size=(3, 3), strides=(1, 1), padding="valid")(d)

        return d

    def back_forward(self, in_video):

        d_0 = self.u_net(in_video[:, :, :, 0:3])

        d_1 = self.u_net(in_video[:, :, :, 1:4])

        d_2 = self.u_net(in_video[:, :, :, 2:5])

        d_3 = self.u_net(tf.keras.layers.Concatenate()([d_0, d_1, d_2]))
        d_3 = BatchNormalization()(d_3)
        d_3 = tf.keras.layers.Activation('relu')(d_3)  # activation doesn't matter for the last layer

        return d_3, d_0, d_1, d_2

    def train(self, video_input_array, DNN_param, video_label_array=None, path_save='./result', name_weights="weights.h5", flag_warm_train=False):
        '''
        Train FastDVDNet.

        Parameters
        ----------
        video_input_array: NDArray
             The video is 4D-numpy (number of batch, image size x, image size y, number of selected frames).
             To prepare this, you can utilize the data handling function of this class.

        DNN_param: dic
            The dictionary is used to define different parameters for DNN. In the following you can see the example of
            this dictionary:

            | DNN_param = {'DNN_batch_size':20, 'epochs': 25, 'shuffle': False, 'validation_split': 0.33}

        video_label_array: NDArray
            The video is 4D-numpy (number of batch, image size x, image size y, number of selected frames).
            To prepare this, you can utilize the data handling function of this class. If it is None, DNN will train
            using the same video as input.

        path_save: str
            The location to a directory where training results can be saved.

        name_weights: str
            The name of the HDF5 file in which the training weights are saved.

        flag_warm_train: bool
            The flag that we can utilize for fine-tuning. If True DNN weights is specified, it is initialized with
            prior save weights; else, it begins with random weights.
        '''

        if video_label_array is None:
            video_label_array = video_input_array

        model_in = tf.keras.layers.Input(
            shape=(video_input_array.shape[1], video_input_array.shape[2], video_input_array.shape[3]))

        r_model, d_0, d_1, d_2 = self.back_forward(model_in)
        r_model = tf.keras.models.Model(inputs=[model_in], outputs=[r_model])

        r_model.compile(loss=['mse'], optimizer='adam')

        logdir = os.path.join(path_save, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0, )

        checkpoint_path = os.path.join(path_save, name_weights)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, save_weights_only=True, verbose=1)

        if flag_warm_train:
            r_model.load_weights(checkpoint_path)
            print("---Loaded model weights from disk!---")

        if self.gpu.gpu_active_flag:
            num_device = str(self.gpu.gpu_device)
            with tf.device('/gpu:'+ num_device):
                print("--- model tarin on GPU " + str(num_device) + "---!")
                r_model.fit(video_input_array, video_label_array[:, :, :, 2:3],
                            batch_size=DNN_param['DNN_batch_size'], epochs=DNN_param['epochs'], shuffle=DNN_param['shuffle'],
                            validation_split=DNN_param['validation_split'], callbacks=[tensorboard_callback, cp_callback])
        else:
            with tf.device('/cpu:0'):
                print('--- model tarin on CPU---!')
                r_model.fit(video_input_array, video_label_array[:, :, :, 2:3],
                            batch_size=DNN_param['DNN_batch_size'], epochs=DNN_param['epochs'],
                            shuffle=DNN_param['shuffle'],
                            validation_split=DNN_param['validation_split'],
                            callbacks=[tensorboard_callback, cp_callback])
            

    def test(self, video_input_array, path_save='./result', name_weights="weights.h5"):
        '''
        Using DNN.

        Parameters
        ----------
        video_input_array: NDArray
             The video is 4D-numpy (number of batch, image size x, image size y, number of selected frames).
             To prepare this, you can utilize the data handling function of this class.

        path_save: str
            The location to a directory where training results can be found.

        name_weights: str
            The name of the HDF5 file in which the training weights are saved.

        Returns
        -------
        video_out_background: NDArray
            The video is 3D-numpy (number of frames, image size x, image size y).
        '''
        checkpoint_path = os.path.join(path_save, name_weights)
        model_in = tf.keras.layers.Input(
            shape=(video_input_array.shape[1], video_input_array.shape[2], video_input_array.shape[3]))

        r_model, d_0, d_1, d_2 = self.back_forward(model_in)
        r_model = tf.keras.models.Model(inputs=[model_in], outputs=[r_model])
        r_model.compile(loss=['mse'], optimizer='adam')
        r_model.load_weights(checkpoint_path)
        print("---Loaded model weights from disk!---")

        if self.gpu.gpu_active_flag:
            num_device = str(self.gpu.gpu_device)
            with tf.device('/gpu:' + num_device):
                print("--- model test on GPU " + str(num_device) + "---!")
                self.predict_array = r_model.predict(video_input_array, verbose=1)

        else:
            with tf.device('/cpu:0'):
                print('--- model test on CPU---!')
                self.predict_array = r_model.predict(video_input_array, verbose=1)

        video_out_background = self.predict_array[:, :, :, 0]

        return video_out_background











