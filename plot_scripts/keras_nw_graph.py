import numpy as np
#from keras.initializers import normal, identity
#from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda
#from keras.optimizers import Adam
from keras import regularizers
#import tensorflow as tf
#import keras.backend as K
from keras.initializers import RandomUniform
from keras.utils import plot_model
import os

# Set the home directory
home_dir = os.path.expanduser('~')

# Set the directory for saving plots
plot_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/plots')

state_size=12
action_dim=2
HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300

# Drawing the critic network
S = Input(shape=[state_size], name='states')
A = Input(shape=[action_dim], name='action2')
w1 = Dense(
           HIDDEN1_UNITS, activation='relu',
           kernel_regularizer=regularizers.l2(0.01),
           kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(state_size), maxval=1.0 / np.sqrt(state_size)),
           bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(state_size), maxval=1.0 / np.sqrt(state_size)),
           name='w1',
           )(S)
a1 = Dense(
           HIDDEN2_UNITS, activation='relu',
           kernel_regularizer=regularizers.l2(0.01),
           kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(action_dim), maxval=1.0 / np.sqrt(action_dim)),
           bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(action_dim), maxval=1.0 / np.sqrt(action_dim)),
           name='a1',
           )(A)
h1 = Dense(
           HIDDEN2_UNITS, activation='relu',
           kernel_regularizer=regularizers.l2(0.01),
           kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS)),
           bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS)),
           name='h1',
           )(w1)
h2 = merge([h1, a1], mode='sum', name='h2')
h3 = Dense(
           HIDDEN2_UNITS, activation='relu',
           kernel_regularizer=regularizers.l2(0.01),
           kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS)),
           bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS)),
           name='h3',
           )(h2)
V = Dense(
          action_dim, activation='linear',  # Not clear what this activation function should be
          kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
          bias_initializer=RandomUniform(minval=-0.003, maxval=0.003),
          name='V')(h3)
model = Model(input=[S, A], output=V)

plot_model(model, to_file=os.path.join(plot_dir,'critic.png'), show_shapes=True)

# Drawing the actor model
S = Input(shape=[state_size])
# Use default initializer to initialize weights
h0 = Dense(HIDDEN1_UNITS, activation='relu',
           kernel_initializer=RandomUniform(minval=-1.0/np.sqrt(state_size), maxval=1.0/np.sqrt(state_size)),
           bias_initializer=RandomUniform(minval=-1.0/np.sqrt(state_size), maxval=1.0/np.sqrt(state_size))
           )(S)
h1 = Dense(HIDDEN2_UNITS, activation='relu',
           kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS)),
           bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS))
           )(h0)
Left_gain_factor = Dense(1,activation='sigmoid',  # to bound output between 0 and 1
                         kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
                         bias_initializer=RandomUniform(minval=-0.003, maxval=0.003))(h1)
Right_gain_factor = Dense(1,activation='sigmoid',  # to bound output between 0 and 1
                          kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
                          bias_initializer=RandomUniform(minval=-0.003, maxval=0.003))(h1)
V = merge([Left_gain_factor,Right_gain_factor],mode='concat')
model = Model(input=S,output=V)

plot_model(model, to_file=os.path.join(plot_dir,'actor.eps'), show_shapes=True)
