import gym
from collections import deque
import tensorflow as tf
from dqn_agent import DQNAgent
import numpy as np
import random

# setting seeds for result reproducibility. This is not super important
random.seed(2212)
np.random.seed(2212)
tf.compat.v1.random.set_random_seed(2212)

class TrainDqn:

    def __init__(self):
        # Hyperparameters / Constants
        self.EPISODES = 300
        self.REPLAY_MEMORY_SIZE = 100000
        self.MINIMUM_REPLAY_MEMORY = 1000
        self.MINIBATCH_SIZE = 32
        self.EPSILON = 1
        self.EPSILON_DECAY = 0.99
        self.MINIMUM_EPSILON = 0.001
        self.DISCOUNT = 0.99
        self.VISUALIZATION = False
        self.ENV_NAME = 'MountainCar-v0'

        # Environment details
        self.env = gym.make(self.ENV_NAME)
        self.action_dim = self.env.action_space.n
        self.observation_dim = self.env.observation_space.shape

        # creating own session to use across all the Keras/Tensorflow models we are using
        self.sess = tf.compat.v1.Session()

        # Replay memory to store experiances of the model with the environment
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

        # Our models to solve the mountaincar problem.
        self.agent = DQNAgent(self.sess, self.action_dim, self.observation_dim)


    def train_dqn_agent(self):
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
        X_cur_states = []
        X_next_states = []
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample
            X_cur_states.append(cur_state)
            X_next_states.append(next_state)
    
        X_cur_states = np.array(X_cur_states)
        X_next_states = np.array(X_next_states)
    
        # action values for the current_states
        cur_action_values = self.agent.model.predict(X_cur_states)
        # action values for the next_states taken from our agent (Q network)
        next_action_values = self.agent.model.predict(X_next_states)
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, next_state, done = sample
            if not done:
                # Q(st, at) = rt + DISCOUNT * max(Q(s(t+1), a(t+1)))
                cur_action_values[index][action] = reward + self.DISCOUNT * np.amax(next_action_values[index])
            else:
                # Q(st, at) = rt
                cur_action_values[index][action] = reward
        # train the agent with new Q values for the states and the actions
        self.agent.model.fit(X_cur_states, cur_action_values, verbose=0)

    def StartPlaying(self):
        max_reward = -999999
        for episode in range(self.EPISODES):
            cur_state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                episode_length += 1
                # set VISUALIZATION = True if want to see agent while training. But makes training a bit slower.
                if self.VISUALIZATION:
                    self.env.render()

                if(np.random.uniform(0, 1) < self.EPSILON):
                    # Take random action
                    action = np.random.randint(0, self.action_dim)
                else:
                    # Take action that maximizes the total reward
                    action = np.argmax(self.agent.model.predict(np.expand_dims(cur_state, axis=0))[0])

                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward

                if done and episode_length < 200:
                    # If episode is ended the we have won the game. So, give some large positive reward
                    reward = 250 + episode_reward
                    # save the model if we are getting maximum score this time
                    if(episode_reward > max_reward):
                        self.agent.model.save_weights(str(episode_reward)+"_agent_.h5")
                else:
                    # In oher cases reward will be proportional to the distance that car has travelled 
                    # from it's previous location + velocity of the car
                    reward = 5*abs(next_state[0] - cur_state[0]) + 3*abs(cur_state[1])
            
                # Add experience to replay memory buffer
                self.replay_memory.append((cur_state, action, reward, next_state, done))
                cur_state = next_state
        
                if(len(self.replay_memory) < self.MINIMUM_REPLAY_MEMORY):
                    continue
        
                self.train_dqn_agent()


            if(self.EPSILON > self.MINIMUM_EPSILON and len(self.replay_memory) > self.MINIMUM_REPLAY_MEMORY):
                self.EPSILON *= self.EPSILON_DECAY

            # some bookkeeping.
            max_reward = max(episode_reward, max_reward)
            print('Episode', episode, 'Episodic Reward', episode_reward, 'Maximum Reward', max_reward, 'EPSILON', self.EPSILON)

obj = TrainDqn()
obj.StartPlaying()