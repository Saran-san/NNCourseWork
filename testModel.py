
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

model_weight_file = "167.0_agent_.h5"

sess = tf.compat.v1.Session()
K.set_session(sess)
env = gym.make('MountainCar-v0')

action_dim = env.action_space.n
observation_dim = env.observation_space.shape

# create and load weights of the model
agent = DQNAgent(sess, action_dim, observation_dim)
agent.model.load_weights(model_weight_file)
# Number of episodes in which agent manages to won the game before time is over
episodes_won = 0
# Number of episodes for which we want to test the agnet
TOTAL_EPISODES = 10 


episodes = []
images = []
for _ in range(TOTAL_EPISODES):
    cur_state = env.reset()
    done = False
    episode_len = 0
    while not done:
        #env.render()
        images.append(env.render("rgb_array"))
        episode_len += 1
        next_state, reward, done, _ = env.step(np.argmax(agent.model.predict(np.expand_dims(cur_state, axis=0))))
        if done and episode_len < 200:
            episodes_won += 1
        cur_state = next_state
    episodes.append(images)
    images = []
     
print(episodes_won, 'EPISODES WON AMONG', TOTAL_EPISODES, 'EPISODES')

i = 1
for frames in episodes:
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    animate = lambda i: patch.set_data(frames[i])
    ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
    HTML(ani.to_jshtml())
    name = "Animation" + model_weight_file + str(i) + ".gif";
    ani.save(name, dpi=80, writer='imagemagick', fps=60)
    i += 1