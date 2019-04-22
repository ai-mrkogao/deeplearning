import tensorflow as tf
import numpy as np
import pandas as pd

import os
import glob
import re

import mnist_data
import plot_utils
import gzip

import matplotlib.pyplot as plt

###
# read encoded stock
###
def readData(filelists):

    try:
        savedir = './data'

        mergeddata = []
        for file in filelists:
            filename = file
            stocksavedir = savedir+'/'+filename

            np_data = []
            for _inputname in ['np_train','np_test']:
                with gzip.open( stocksavedir+'/'+_inputname+'.npy.gz', 'r') as infile:
                    np_data.append(np.load(infile))

            df_data = []
            for _inputname in ['df_train','df_test']:
                df_data.append(pd.read_pickle(stocksavedir+'/'+_inputname+'.pkl'))

            mergeddata.append([np_data[0],np_data[1],df_data[0],df_data[1]])

        return mergeddata

    except Exception as e:
        print(e)

###
# read org stock data
###
def readStock(filelists):
    savedir = './data/hist'

    dfstocks = []
    for file in filelists:
        filepath = savedir+'/'+file
        dfstocks.append(pd.read_pickle(filepath))
    return dfstocks


###
# decode
###

def decode_npdata(np_data,max_seq_len,df_stock,df_data):

    curindex = max_seq_len*4+1

    start_datetimeindex = df_data.index[0]
    start_index = df_stock[:start_datetimeindex].shape[0]

    decoded_prices = {}
    decoded_prices['Open'] = []
    decoded_prices['High'] = []
    decoded_prices['Low'] = []
    decoded_prices['Close'] = []
    decoded_prices['CurOpen'] = []

    for i in range(np_data.shape[0]):
        p_open,p_high,p_low,p_close,p_curopen = np_data[i][curindex-5:curindex]

        for _p,_col in zip([p_open,p_high,p_low,p_close],['Open','High','Low','Close']):
            _min = df_stock[_col][start_index+i - (max_seq_len+1):start_index+i].min()
            _max = df_stock[_col][start_index+i - (max_seq_len+1):start_index+i].max()

            decoded_p = _p*((_max - _min)+ 1e-7) + _min
            decoded_prices[_col].append(decoded_p)

        for _p,_col in zip([p_curopen],['Open']):
            print("df_data {} df_stock {}".format(df_data.index[i],df_stock.index[start_index+i-1]))
            _min = df_stock[_col][start_index+i - (max_seq_len+1):start_index+i].min()
            _max = df_stock[_col][start_index+i - (max_seq_len+1):start_index+i].max()

            decoded_p = _p*((_max - _min)+ 1e-7) + _min
            decoded_prices['CurOpen'].append(decoded_p)

    return decoded_prices



#####
# variational autoencoder
####
class vae:
    def __init__(self,session,learning_rate,dense_layers,dense_funcs,dim_z,dim_out):

        self.session = session
        self.dim_in = dim_out
        self.dim_out = dim_out
        self.dim_z = dim_z

        self.lrate = learning_rate
        self.dense_layers = dense_layers
        self.dense_funcs = dense_funcs

        self.model_name = 'vae'
        self.logs_dir = './model'
        self._build_network()

    def _build_network(self):

        self.x_hat = tf.placeholder(tf.float32, shape=[None, self.dim_in], name='input_denoising')
        self.x = tf.placeholder(tf.float32, shape=[None, self.dim_out], name='target')

        # dropout
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # input for PMLR
        self.z_in = tf.placeholder(tf.float32, shape=[None, self.dim_z], name='latent_variable')

        self.encoder_dense_layers = self.dense_layers+[self.dim_z]
        self.encoder_dense_funcs = self.dense_funcs
        print("encoder_dense_layers {}".format(self.encoder_dense_layers))

        self.mu,self.sigma = self.encoder(self.x_hat,self.encoder_dense_layers,self.encoder_dense_funcs,self.keep_prob)

        print("mu:{} sigma:{}".format(self.mu.shape,self.sigma.shape))
        # sampling by re-parameterization technique
        self.z = self.mu + self.sigma * tf.random_normal(tf.shape(self.mu), 0, 1, dtype=tf.float32)
        print("encoder output z:{}".format(self.z.shape))

        self.decoder_dense_layers = self.dense_layers[::-1]+[self.dim_out]
        self.decoder_activ = self.encoder_dense_funcs[::-1]
        print("decoder_dense_layers {}".format(self.decoder_dense_layers))

        # decoding
        self.y = self.decoder(self.z, self.decoder_dense_layers,self.decoder_activ, self.keep_prob)
        self.y = tf.clip_by_value(self.y, 1e-8, 1 - 1e-8)
        print("decoder output y:{}".format(self.y.shape))
        # loss
        self.marginal_likelihood = tf.reduce_sum(self.x * tf.log(self.y) + (1 - self.x) * tf.log(1 - self.y), 1)
        self.KL_divergence = 0.5 * tf.reduce_sum(tf.square(self.mu) + tf.square(self.sigma) - tf.log(1e-8 + tf.square(self.sigma)) - 1, 1)

        self.marginal_likelihood = tf.reduce_mean(self.marginal_likelihood)
        self.KL_divergence = tf.reduce_mean(self.KL_divergence)

        self.ELBO = self.marginal_likelihood - self.KL_divergence

        self.loss = -self.ELBO

        self.decoded = self.sample_decoder(self.z_in, self.dim_out)

        self.train_op = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)
        self.neg_marginal_likelihood = -1*self.marginal_likelihood

        self.global_step = tf.Variable(0, name='global_step',trainable=False)
        self.saver = tf.train.Saver(max_to_keep=1)


    def save(self, step):
        model_name = self.model_name + ".model"

        model_logs_dir = self.logs_dir
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)

        self.saver.save(
            self.session,
            os.path.join(model_logs_dir, model_name),
            global_step=step
        )
    def load(self):

        ckpt = tf.train.get_checkpoint_state(self.logs_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # print('ckpt_name {} self.logs_dir {}'.format(ckpt_name,self.logs_dir))
            self.saver.restore(self.session, os.path.join(self.logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a " +self.logs_dir+ " checkpoint")
            return False, 0

    def sample_decoder(self,z, dim_img):

        # decoding
        y = self.decoder(z, self.decoder_dense_layers,self.decoder_activ, 1.0, reuse=True)
        return y

    def encoder(self,x, dense_layers,dense_func,keep_prob):
        with tf.variable_scope("encoder"):

            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # dense layers
            dense_input_shape = x.get_shape()[1]
            dense_input = x
            for _i,_n_hidden in enumerate(dense_layers[:-1]):
                net_w = tf.get_variable('w'+str(_i),[dense_input_shape,_n_hidden],initializer=w_init)
                net_b = tf.get_variable('b'+str(_i),[_n_hidden],initializer = b_init)
                net_h = tf.matmul(dense_input,net_w) + net_b
                if dense_func[_i] == 'elu':
                    actfunc = tf.nn.elu
                elif dense_func[_i] == 'tanh':
                    actfunc = tf.nn.tanh
                net_h = actfunc(net_h)
                net_h = tf.nn.dropout(net_h,keep_prob)
                dense_input_shape = net_h.get_shape()[1]
                dense_input = net_h

            # final output
            final_w = tf.get_variable("w_final",[dense_input.get_shape()[1],dense_layers[-1]*2],initializer = w_init)
            final_b = tf.get_variable("b_final",[dense_layers[-1]*2],initializer=b_init)
            gaussian_params = tf.matmul(dense_input,final_w) + final_b

            mean = gaussian_params[:,:dense_layers[-1]]
            stddev = 1e-6 +tf.nn.softplus(gaussian_params[:,dense_layers[-1]:])
            print("gaussian encoder mean:{} ".format(mean.shape))


        return mean, stddev

    # Bernoulli as decoder
    # bernoulli_decoder(z, [500,500,784],['tanh','elu'], keep_prob, reuse=False)
    def decoder(self,z, dense_layers,dense_func, keep_prob, reuse=False):

        with tf.variable_scope("decoder", reuse=reuse):
            # initializers
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer(0.)

            # dense layers
            dense_input_shape = z.get_shape()[1]
            dense_input = z
            for _i,_n_hidden in enumerate(dense_layers[:-1]):
                net_w = tf.get_variable('w'+str(_i),[dense_input_shape,_n_hidden],initializer=w_init)
                net_b = tf.get_variable('b'+str(_i),[_n_hidden],initializer = b_init)
                net_h = tf.matmul(dense_input,net_w) + net_b
                if dense_func[_i] == 'elu':
                    actfunc = tf.nn.elu
                elif dense_func[_i] == 'tanh':
                    actfunc = tf.nn.tanh
                net_h = actfunc(net_h)
                net_h = tf.nn.dropout(net_h,keep_prob)
                dense_input_shape = net_h.get_shape()[1]
                dense_input = net_h

            # final output
            final_w = tf.get_variable("w_final",[dense_input.get_shape()[1],dense_layers[-1]],initializer = w_init)
            final_b = tf.get_variable("b_final",[dense_layers[-1]],initializer=b_init)
            y = tf.sigmoid(tf.matmul(dense_input,final_w)+final_b)


        return y



def save_scattered_image(z, id, name='scattered_image.jpg'):
    N = 4
    z_range = 3
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range-2, z_range+2])
    axes.set_ylim([-z_range-2, z_range+2])
    plt.grid(True)
    plt.savefig("./results/time/" + name)


def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
