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
        self.noOfEpisodes = 300
        self.ReplayMemoryQueueSize = 100000
        self.minReplayMemoryQueueSize = 1000
        self.sampleBatchSize = 32
        self.epsilon = 1
        self.epsilonDecay = 0.99
        self.minEpsilon = 0.001
        self.discount = 0.99
        self.doRender = False
        self.gameEnv = 'MountainCar-v0'

        # Environment details
        self.env = gym.make(self.gameEnv)
        self.action_dim = self.env.action_space.n
        self.observation_dim = self.env.observation_space.shape

        # creating own session to use across all the Keras/Tensorflow models we are using
        self.sess = tf.compat.v1.Session()

        # Replay memory to store experiances of the model with the environment
        self.replay_memory = deque(maxlen=self.ReplayMemoryQueueSize)

        # Our models to solve the mountaincar problem.
        self.agent = DQNAgent(self.sess, self.action_dim, self.observation_dim)


    def train_dqn_agent(self):
        minibatch = random.sample(self.replay_memory, self.sampleBatchSize)
        currStates = []
        nextStates = []
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, nextState, done = sample
            currStates.append(cur_state)
            nextStates.append(nextState)
    
        currStates = np.array(currStates)
        nextStates = np.array(nextStates)
    
        # action values for the current_states
        currActionValues = self.agent.model.predict(currStates)
        # action values for the next_states taken from our agent (Q network)
        nextActionValues = self.agent.model.predict(nextStates)
        for index, sample in enumerate(minibatch):
            cur_state, action, reward, nextState, done = sample
            if not done:
                # Q(st, at) = rt + discount * max(Q(s(t+1), a(t+1)))
                currActionValues[index][action] = reward + self.discount * np.amax(nextActionValues[index])
            else:
                # Q(st, at) = rt
                currActionValues[index][action] = reward
        # train the agent with new Q values for the states and the actions
        self.agent.model.fit(currStates, currActionValues, verbose=0)

    def StartPlaying(self):
        max_reward = -999999
        for episode in range(self.noOfEpisodes):
            cur_state = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                episode_length += 1
                # set doRender = True if want to see agent while training. But makes training a bit slower.
                if self.doRender:
                    self.env.render()

                if(np.random.uniform(0, 1) < self.epsilon):
                    # Take random action
                    action = np.random.randint(0, self.action_dim)
                else:
                    # Take action that maximizes the total reward
                    action = np.argmax(self.agent.model.predict(np.expand_dims(cur_state, axis=0))[0])

                nextState, reward, done, _ = self.env.step(action)

                episode_reward += reward

                if done and episode_length < 200:
                    # If episode is ended the we have won the game. So, give some large positive reward
                    reward = 250 + episode_reward
                    # save the model if we are getting maximum score this time
                    if(episode_reward > max_reward):
                        self.agent.model.save_weights(str(episode_reward)+"_agent_.h5")
                else:
                    # In other cases reward will be proportional to the distance that car has travelled 
                    # from it's previous location + velocity of the car
                    reward = 5*abs(nextState[0] - cur_state[0]) + 3*abs(cur_state[1])
            
                # Add experience to replay memory buffer
                self.replay_memory.append((cur_state, action, reward, nextState, done))
                cur_state = nextState
        
                if(len(self.replay_memory) < self.minReplayMemoryQueueSize):
                    continue
        
                self.train_dqn_agent()


            if(self.epsilon > self.minEpsilon and len(self.replay_memory) > self.minReplayMemoryQueueSize):
                self.epsilon *= self.epsilonDecay

            # some bookkeeping.
            max_reward = max(episode_reward, max_reward)
            print('Episode', episode, 'Episodic Reward', episode_reward, 'Maximum Reward', max_reward, 'epsilon', self.epsilon)

obj = TrainDqn()
obj.StartPlaying()