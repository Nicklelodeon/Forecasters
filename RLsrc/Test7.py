from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from State import State

env = State()
env.create_state([-1,0,0])
env.show_network()
env.observation_space.sample()
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        print(type(action))
        print("action" + str(action))
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from rl.agents import DDPGAgent
def build_model(states, actions):
    model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Dense(64, activation='relu', input_shape=states),
    tensorflow.keras.layers.Dense(64, activation='relu'),
    #tensorflow.keras.layers.Flatten(),
    tensorflow.keras.layers.Dense(2, activation='softmax')
])
    return model
states = env.observation_space.shape
actions = env.action_space
states
actions
env.action_space.shape
model = build_model(states, actions)

model.summary()
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn
def build_agent2(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    ddpg = DDPGAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return ddpg
dqn = build_agent(model, 2)
#dqn.compile(optimizer=Adam(learning_rate=1e-3), metrics=['mae'])
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)






scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))