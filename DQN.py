import gym
import math
import random
import signal
import sys
import traceback

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import *

from SumTree import SumTree

MODEL_PATH = 'CartPole-v0.h5'

class Brain:

    def __init__(self, state_size, action_size, load_model=False):
        self.state_size = state_size
        self.action_size = action_size

        if load_model:
            self.model = self._loadModel()
        else:
            self.model = self._createModel()
            self.target_model = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(output_dim=64, activation='relu', input_dim=state_size))
        model.add(Dense(output_dim=action_size, activation='linear'))

        opt = Adam()
        model.compile(loss='mse', optimizer=opt)

        return model

    def _loadModel(self):
        return load_model(MODEL_PATH)

    def train(self, x, y, epoch=1, verbose=False):
        self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)

    def predict(self, x, target=False):
        if target:
            return self.target_model.predict(x)
        else:
            return self.model.predict(  x)

    def predictOne(self, x, target=False):
        return self.predict(x.reshape(1, self.state_size), target).flatten()

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())


class Memory:

    eps = 0.01  # Small positive constant to ensure non-zero priority.
    a = 0.6  # Controls difference between high and low priority.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.eps) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            start_sum = segment * i
            end_sum = segment * (i + 1)

            s = random.uniform(start_sum, end_sum)
            index, p, data = self.tree.get(s)
            batch.append((index, data))

        return batch

    def update(self, index, error):
        p = self._getPriority(error)
        self.tree.update(index, p)


MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MIN_EPSILON = 0.1
MAX_EPSILON = 1.00

LAMBDA = 0.001

UPDATE_TARGET_FREQUENCY = 5000

class Agent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.brain = Brain(state_size, action_size)
        self.memory = None  # Expecting memory from random agent.

    def act(self, s):   
        if random.random() < self.epsilon:
            return random.randint(0, action_size - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def observe(self, sample):
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            print('-----Updated target network-----')
            self.brain.updateTargetModel()

        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        batch_len = len(batch)
        no_state = np.zeros(self.state_size)

        state = np.array([o[1][0] for o in batch])
        state_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = self.brain.predict(state)

        p_ = self.brain.predict(state_)
        pTarget_ = self.brain.predict(state_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            s, a, r, s_ = batch[i][1]

            t = p[i]
            oldVal = t[a]

            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ np.argmax(p_[i]) ]

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])  

        return x, y, errors

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        # Update errors in memory.
        for i in range(len(batch)):
            index = batch[i][0]
            self.memory.update(index, errors[i])

        self.brain.train(x, y)

        

class PlayAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.brain = Brain(state_size, action_size, load_model=True)

    def act(self, s):   
        return np.argmax(self.brain.predictOne(s))

    def observe(self, s):
        pass

    def replay(self):
        pass

class RandomAgent:

    def __init__(self, action_size):
        self.action_size = action_size

        self.steps = 0
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        return random.randint(0, self.action_size - 1)
    
    def observe(self, s):
        error = abs(s[2])

        self.memory.add(error, s)
        self.steps += 1

    def replay(self):
        pass

    def memoryFull(self):
        return self.steps >= MEMORY_CAPACITY


class Environment:

    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem).env

        self.episodes = 0

        self.rewards = []
        self.startQFunc = []

    def run(self, agent, render=True, trackStats=False):
        s = self.env.reset()
        R = 0

        while True:
            if render:
                self.env.render()

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()          

            s = s_
            R += r

            if done:
                break

        if trackStats:
            self.rewards.append(r)

            zero_state = np.zeros(self.env.observation_space.shape[0])
            self.startQFunc.append(agent.brain.predictOne(zero_state))


        self.episodes += 1

        print(f'Episode {self.episodes} rewards: {R}')


TRAIN_AND_SAVE_MODEL = True
TRACK_STATS = True
TEST_MODEL = True


if __name__ == '__main__':

    gym_problem = 'CartPole-v0'
    env = Environment(gym_problem)

    state_size = env.env.observation_space.shape[0]
    action_size = env.env.action_space.n

    if TRAIN_AND_SAVE_MODEL:
        agent = Agent(state_size, action_size)
        random_agent = RandomAgent(action_size)

        while not random_agent.memoryFull():
            env.run(random_agent, render=False)
        
        agent.memory = random_agent.memory
        random_agent = None
        env.episodes = 0

    elif TEST_MODEL:
        agent = PlayAgent(state_size, action_size)
    else:
        print('No train/test mode set.')
        sys.exit(0)

    def SIGTSTP_handler(x,y):
        print ('Keyboard interrupt')

        if TRAIN_AND_SAVE_MODEL:
            print('Saving model')
            agent.brain.model.save(MODEL_PATH)

            if TRACK_STATS:
                print('Saving stats.')

                import pickle
                pickle.dump((env.rewards, env.startQFunc), open( "stats.p", "wb" ))

        sys.exit(0)

    signal.signal(signal.SIGTSTP, SIGTSTP_handler)

    try:
        while True:
            env.run(agent, trackStats=TRACK_STATS)
    except Exception as e:
        print(traceback.format_exc())
    finally:
        if TRAIN_AND_SAVE_MODEL:
            print('Saving model')
            agent.brain.model.save(MODEL_PATH)

        sys.exit()

    