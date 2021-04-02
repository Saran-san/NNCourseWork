
import gym
from dqn_agent import DQNAgent
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from IPython.display import HTML

modelWeightFile = "167.0_agent_.h5"

sess = tf.compat.v1.Session()
K.set_session(sess)
env = gym.make('MountainCar-v0')

actionDim = env.action_space.n
observationDim = env.observation_space.shape

# create and load weights of the model
agent = DQNAgent(sess, actionDim, observationDim)
agent.model.load_weights(modelWeightFile)
# Number of episodes won in total episodes
episodesWon = 0
# Number of episodes to test the agent
TOTAL_EPISODES = 10 


episodes = []
images = []
for _ in range(TOTAL_EPISODES):
    currState = env.reset()
    done = False
    episodeLen = 0
    while not done:
        #env.render()
        images.append(env.render("rgb_array"))
        episodeLen += 1
        next_state, reward, done, _ = env.step(np.argmax(agent.model.predict(np.expand_dims(currState, axis=0))))
        if done and episodeLen < 200:
            episodesWon += 1
        currState = next_state
    episodes.append(images)
    images = []
     
print(episodesWon, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')

i = 1
for frames in episodes:
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    animate = lambda i: patch.set_data(frames[i])
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
    HTML(ani.to_jshtml())
    name = "Animation" + modelWeightFile + str(i) + ".gif";
    ani.save(name, dpi=80, writer='imagemagick', fps=60)
    i += 1