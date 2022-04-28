import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import os

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__(name=name)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, self.model_name)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = self.fc1(tf.concat[state, action], axis=1)
        q1_action_value = self.fc2(q1_action_value)
        q = self.q(q1_action_value)

        return q

    
class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, num_actions, name, chkpt_dir='tmp/td3'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.num_actions = num_actions
        self.model_name = name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, name)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation='tanh')

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu