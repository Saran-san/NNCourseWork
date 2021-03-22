import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# setting seeds for result reproducibility. This is not super important
tf.compat.v1.random.set_random_seed(1024)

class DQNAgent:
    def __init__(self, sess, action_dim, observation_dim):
        # Force keras to use the session that we have created
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.model = self.create_model()

    def create_model(self):
        state_input = Input(shape=(self.observation_dim))
        state_h1 = Dense(256, activation='relu')(state_input)
        state_h2 = Dense(64, activation='relu')(state_h1)
        state_h3 = Dense(256, activation='relu')(state_h2)
        state_h4 = Dense(64, activation='relu')(state_h3)
        output = Dense(self.action_dim, activation='linear')(state_h4)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam())
        return model

