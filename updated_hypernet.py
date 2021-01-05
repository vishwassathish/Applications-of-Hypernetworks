# Dimi's updated implementation of the cosine hypernetwork

import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt;
from tensorflow.keras.layers import Layer
tf.disable_eager_execution()


import PIL.Image as Image
h1 = 16;
h2 = 16;

# The total number of parameters to predict from hyper model must be predetermined
PARAMS = 1*h1 + h1 + h1*h2 + h2 + h2*1 + 1     # kernel weights and bias terms for the dense layers in inference model

# This model creates the weights of inference model

out_H = out_W = 32;


time = tf.placeholder(tf.float32,shape=(None,1),name='time');
freq = tf.placeholder(tf.float32,shape=(None,out_H*out_W),name='freq');
y = tf.placeholder(tf.float32,shape=(None,1),name='y');


h_sz = 64;
hyper1 = tf.keras.layers.Dense(h_sz, activation='elu', name='dense_1')
hyper2 = tf.keras.layers.Dense(h_sz, activation='elu', name='dense_2')
hyper3 = tf.keras.layers.Dense(h_sz, activation='elu', name='dense_3')
hyper4 = tf.keras.layers.Dense(PARAMS, name='dense_4')



def hyper(x):
        h = hyper1(x);
        h = hyper2(h)
        h = hyper3(h);
        h = hyper4(h);
        return h;
        
        


def modulated_network(weights,t):

    last_used = 0
    num_w = 1*h1;
    w1 = tf.reshape(weights[:,last_used:last_used+num_w], [-1,1,h1])
    last_used += num_w
    
    num_w = 1*h1;
    b1 = tf.reshape(weights[:,last_used:last_used+num_w], [-1,1,h1])
    last_used += num_w
    
    
    
    num_w = h1*h2;
    w2 = tf.reshape(weights[:,last_used:last_used+num_w], [-1,h1,h2])
    last_used += num_w
    
    num_w = 1*h2;
    b2 = tf.reshape(weights[:,last_used:last_used+num_w], [-1,1,h2])
    last_used += num_w
    
    
    
    num_w = h2*1;
    w3 = tf.reshape(weights[:,last_used:last_used+num_w], [-1,h2,1])
    last_used += num_w
    
    num_w = 1*1;
    b3 = tf.reshape(weights[:,last_used:last_used+num_w], [-1,1,1])
    last_used += num_w
    
    t = tf.reshape(t,[-1,1,1]);
    
    print(w1.shape)
    print(w2.shape)
    print(w3.shape)
    print(t.shape)
    h = tf.matmul(t,w1)+b1;
    h = tf.nn.elu(h);
    print(h.shape)
    h = tf.matmul(h,w2)+b2;
    h = tf.nn.elu(h);
    print(h.shape)

    h = tf.matmul(h,w3)+b3;
    print(h.shape)

    
    y = tf.squeeze(h,axis=1);
    print(y.shape)
    
    return y;



weights = hyper(freq);
y_hat = modulated_network(weights,time);


loss = tf.reduce_mean(tf.square(y-y_hat))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4);
train_op = optimizer.minimize(loss,var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''))




f_low = 1;
f_high = 8;
batch_size = 32;


viz_period = 500;
hist_size = 1e3;
hist_loss = [];

freq_I = np.zeros((8,out_H*out_W))
for idf in range(8):
        f = idf + 1;
        t = np.linspace(0,1.0,100);
        yy = np.sin(2*np.pi*f*t);


        plt.plot(t,yy)
        plt.axis('off')

        fnm = 'cos.png'

        plt.savefig(fnm,bbox_inches='tight')
        plt.clf()

        I = np.asarray(Image.open(fnm))


        def rgb2gray(rgb):
                return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
        I = rgb2gray(I/255.0)

        h_border = 10;
        w_border = 10;
        I = I[h_border:-h_border,w_border:-w_border];


        I = Image.fromarray(I).resize((out_H,out_W),resample=Image.BICUBIC);
        I = np.asarray(I);
        freq_I[idf,:] = I.reshape([1,-1]);

        plt.imshow(I,cmap='gray')
        plt.savefig('pix_cos'+str(idf)+'.png',bbox_inches='tight')
        plt.clf()
                        
with tf.Session() as sess:

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        
        for minibatch in range(5000000):
        
                      
                idx = np.random.randint(0,8,batch_size)
                batch_I = freq_I[idx,:];
                batch_t = np.float32((np.random.randint(0,100,batch_size)/100).reshape([-1,1]))
                batch_y = np.sin(2*np.pi*(idx+1).reshape([-1,1])*batch_t)
                

                _,LOSS = sess.run([train_op,loss],feed_dict={time:batch_t,freq:batch_I,y:batch_y})
                if minibatch%viz_period == 0:
                    print("Loss {:2}: {:2.3f}".format(minibatch, LOSS))
                    hist_loss += [LOSS];

                    comb = np.float32(np.linspace(0,1,200).reshape([-1,1]));
                    
                    idx = np.random.randint(0,8,1)
                    ground_f = np.float32(idx)+1;
                    
                    batch_I = freq_I[idx,:].reshape([1,-1]);
                    batch_I = np.tile(batch_I,[comb.shape[0],1])
                    [SIN_HAT] = sess.run([y_hat],feed_dict={time:comb,freq:batch_I})
                    
                    SIN = np.sin(2*np.pi*ground_f*comb);
                    
                    plt.plot(comb,SIN_HAT)
                    plt.plot(comb,SIN,color='r')
                    plt.savefig('./sin_hat.png')
                    plt.clf()
                    
                    plt.plot(np.array(hist_loss))
                    plt.savefig('./loss.png')
                    plt.clf()
                    
                    if len(hist_loss)>hist_size:
                            hist_loss = hist_loss[-hist_size:];
