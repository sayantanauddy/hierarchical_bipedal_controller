import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, merge
from keras.optimizers import Adam
from keras import regularizers
import tensorflow as tf
import keras.backend as K
from keras.initializers import RandomUniform
from collections import deque
import random
import json
import gym
import os

from matsuoka_walk import Logger, log

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 300

# Test the policy after every n episodes
TEST_AFTER_N_EPISODES = 10

# Size of replay buffer
BUFFER_SIZE = 100000

# Mini-batch size
BATCH_SIZE = 64

# Discount factor
GAMMA = 0.99

# Target Network HyperParameter
TAU = 0.001

# Learning rate for Actor
LRA = 0.0001

# Lerning rate for Critic
LRC = 0.001

# Factor controlling reduction of influence of exploration noise over time
EXPLORE = 100000.

# Maximum episodes for training
train_episode_count = 1000

# Episode count for testing
test_episode_count = 100

# Maximum steps in each episode
max_steps = 40

# Dimension of action and state spaces
action_dim = 2  # left_gain_factor, right_gain_factor
state_dim = 12  # alpha, beta, gamma, d_alpha, d_beta, d_gamma, x, y, z, d_x, d_y, d_z

# Seed is used for reproducing the same results
SEED_FOR_RANDOM = 456

# Set the home directory
home_dir = os.path.expanduser('~')

# Directory to save trained models
model_dir = os.path.join(home_dir, 'computing/repositories/hierarchical_bipedal_controller/ddpg_trained_models')


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        log('[DDPG] Building the actor model')
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
        return model, model.trainable_weights, S


class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size

        K.set_session(sess)

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size, action_dim):
        log('[DDPG] Building the critic model')
        S = Input(shape=[state_size])
        A = Input(shape=[action_dim], name='action2')
        w1 = Dense(HIDDEN1_UNITS, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(state_size), maxval=1.0 / np.sqrt(state_size)),
                   bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(state_size), maxval=1.0 / np.sqrt(state_size))
                   )(S)
        a1 = Dense(HIDDEN2_UNITS, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(action_dim), maxval=1.0 / np.sqrt(action_dim)),
                   bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(action_dim), maxval=1.0 / np.sqrt(action_dim))
                   )(A)
        h1 = Dense(HIDDEN2_UNITS, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS)),
                   bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN1_UNITS), maxval=1.0 / np.sqrt(HIDDEN1_UNITS))
                   )(w1)
        h2 = merge([h1, a1], mode='sum')
        h3 = Dense(HIDDEN2_UNITS, activation='relu',
                   kernel_regularizer=regularizers.l2(0.01),
                   kernel_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN2_UNITS), maxval=1.0 / np.sqrt(HIDDEN2_UNITS)),
                   bias_initializer=RandomUniform(minval=-1.0 / np.sqrt(HIDDEN2_UNITS), maxval=1.0 / np.sqrt(HIDDEN2_UNITS))
                   )(h2)
        V = Dense(action_dim, activation='linear',  # Linear activation function
                  kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
                  bias_initializer=RandomUniform(minval=-0.003, maxval=0.003))(h3)
        model = Model(input=[S, A], output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)


class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


OU = OU()


def deviation_controller(train_indicator=0, identifier=''):    #1 means Train, 0 means simply Run

    # np.random.seed(1337)

    # The train_indicator is switched internally to test the model after every n runs
    # So a separate flag indicates if the entire run is a test run, in which case the train_indicator always stays 0
    only_test_run = False
    if train_indicator == 0:
        log('[DDPG TEST ] This is a test run')
        only_test_run = True
    else:
        log('[DDPG] This is a training run')

    done = False
    step = 0
    epsilon = 1

    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # Create the actor and acritic models and the replay buffer
    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)  # Create replay buffer

    # Register the matsuoka environment
    ENV_NAME = 'matsuoka_env-v0'
    gym.undo_logger_setup()
    env = gym.make(ENV_NAME)
    env._max_episode_steps = max_steps
    np.random.seed(SEED_FOR_RANDOM)
    env.seed(SEED_FOR_RANDOM)

    # Load existing weights if this is a test run
    if only_test_run:
        log('[DDPG TEST ] Loading existing weights')
        try:
            actor.model.load_weights(os.path.join(model_dir, 'actormodel_'+identifier+'.h5'))
            critic.model.load_weights(os.path.join(model_dir, 'criticmodel_'+identifier+'.h5'))
            actor.target_model.load_weights(os.path.join(model_dir, 'actormodel_'+identifier+'.h5'))
            critic.target_model.load_weights(os.path.join(model_dir, 'criticmodel_'+identifier+'.h5'))
            log('[DDPG TEST ] Weight load successfully')
        except:
            print("Cannot find the weight")

    # This flag indicates if a test has just been done
    just_tested = False

    # Counter for episodes
    i = 1

    # While max number of episodes is not over
    episode_count = train_episode_count if train_indicator == 1 else test_episode_count
    log('[DDPG ' + ('' if train_indicator else 'TEST') + '] Number of max episodes: {}'.format(episode_count))

    while i<=episode_count:

        # Test the policy after every n episodes
        # So after episode 20 completes, i will be 21 and the if will evaluate to True
        # If train_indicator is initially set to 0, then execute the else block only
        # This logic of switching the train_indicator is only needed during a training run
        if not only_test_run:
            if not just_tested and (i - 1) > 0 and ((i-1) % TEST_AFTER_N_EPISODES == 0):
                train_indicator = 0
                # We are testing for the last episode
                i -= 1
                just_tested = True
                log('[DDPG TEST] Testing network after episode {}'.format(i))
            else:
                train_indicator = 1
                just_tested = False

        log('[DDPG ' + ('' if train_indicator else 'TEST') + '] Episode : ' + str(i) + ' Replay Buffer ' + str(buff.count()))

        ob = env.reset()

        s_t = ob

        total_reward = 0.

        for j in range(max_steps):

            loss = 0
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            # Include noise only during training
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.15, 0.2)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.0, 0.15, 0.2)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]

            # Step the environment and fetch the observation, reward and terminal_flag
            ob, r_t, done, info = env.step(a_t[0])

            # Set the new state
            s_t1 = ob

            # Add to replay buffer
            buff.add(s_t, a_t[0], r_t, s_t1, done)

            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])

            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * target_q_values[k]

            if train_indicator:
                log('[DDPG] Updating the models')
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            log('[DDPG ' + ('' if train_indicator else 'TEST') + '] Episode: {0} Step: {1} Action: {2} Reward: {3} Loss: {4}'.format(i, step, a_t, r_t, loss))

            step += 1
            if done:
                break

            # Save the model after every n episodes
            if i > 0 and np.mod(i, TEST_AFTER_N_EPISODES) == 0:
                if (train_indicator):
                    log('[DDPG] Saving the model')
                    actor.model.save_weights(os.path.join(model_dir, 'actormodel_'+identifier+'_{}'.format(i)+'.h5'), overwrite=True)
                    with open(os.path.join(model_dir, 'actormodel_'+identifier+'_{}'.format(i)+'.json'), "w") as outfile:
                        json.dump(actor.model.to_json(), outfile)

                    critic.model.save_weights(os.path.join(model_dir, 'criticmodel_'+identifier+'_{}'.format(i)+'.h5'), overwrite=True)
                    with open(os.path.join(model_dir, 'criticmodel_'+identifier+'_{}'.format(i)+'.json'), "w") as outfile:
                        json.dump(critic.model.to_json(), outfile)

        # Reinitialize step count after an episode is done
        step = 0

        log('[DDPG ' + ('' if train_indicator else 'TEST') + '] TOTAL REWARD @ ' + str(i) + '-th Episode  : Reward ' + str(total_reward))
        log('')

        # Increment the episode count
        i += 1

    env.close()
    log('[DDPG] Finish')


if __name__ == "__main__":
    from matsuoka_walk.matsuoka_env import MatsuokaEnv
    from gym.envs.registration import register

    register(
        id='matsuoka_env-v0',
        entry_point='matsuoka_walk:matsuoka_env.MatsuokaEnv',
        max_episode_steps=40,
    )

    # Set the logging variables
    # This also creates a new log file
    Logger(log_dir=os.path.join(home_dir, '.bio_walk/logs/'), log_flag=True)

    # Identifier used for saving model weights
    identifier = Logger.datetime_str
    log('[DDPG MAIN] Model weight identifier is {}'.format(identifier))

    # Start the DDPG algorithm
    # Set train_indicator=1 for training and train_indicator=0 for testing
    # For testing, set identifier to that of the desired weights to be loaded
    # identifier = '20171027_144930'
    deviation_controller(train_indicator=1, identifier=identifier)
