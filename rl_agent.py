from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env
from absl import app

import rl.core
from gym import spaces

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class MoveToBeacon_KerasRL(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.cnt = 0

    def step(self, obs):
        # super().step(obs)
        if self.cnt == 0:
            self.cnt += 1
            return (5, 5)
        elif self.cnt == 1:
            self.cnt += 1
            return (5, 50)
        elif self.cnt == 2:
            self.cnt += 1
            return (50, 50)
        else:
            self.cnt = 0
            return (50, 5)


class PySC2ToKerasRL_env(rl.core.Env):
    # converts PySC2 env outputs to the inputs Keras-rl agents expect

    def __init__(self, PySC2_env):
        self.env = PySC2_env

        '''Move to Beacon'''
        # define the action space and obs space based on the map, this is the move to beacon
        self.action_space = spaces.Box(np.array(0), np.array(64**2))  #flatten action space

        self.observation_space = spaces.Box(np.zeros(64 ** 2),
                                            np.array([4 for x in range(0, 64 ** 2)]))  # flatten minimap

        self.action_space.n = 64**2 # flatten the action space

    def step(self, step_actions):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        # map x,y (0-63,0-63) coord to the move marine command

        x = step_actions % 64
        y = step_actions // 64

        step_action = [actions.FUNCTIONS.Move_minimap("now", (x,y))]

        try:
            timesteps = getattr(self.env, "step")(step_action)  # step env
        except:
            timesteps = getattr(self.env, "step")([actions.FUNCTIONS.select_army('select')])  # select the marine

        obs = timesteps[0].observation  # retrieve updated obs

        obs = obs.feature_minimap.player_relative  # a 0-63,0-63 array with 0-4 for each value
        obs = obs.flatten()  # map obs to a flattened space

        reward = timesteps[0].reward  # get reward
        done = timesteps[0].last()  # check if last step
        info = {'items':1} # dummy dict entry to make keras rl happy

        return obs, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        getattr(self.env, "reset")()  # reset env using parent method

        reset_action = [actions.FUNCTIONS.select_army('select')]  # select marine
        timesteps = getattr(self.env, "step")(reset_action)  # select the marine

        obs = timesteps[0].observation  # retrieve initial obs
        # format obs
        obs = obs.feature_minimap.player_relative  # a 0-63,0-63 array with 0-4 for each value
        obs = obs.flatten()  # map obs to a flattened space
        return obs

    def render(self, mode='human', close=False):
        pass  # can't really do much here since the env is already set up

    def close(self):
        getattr(self.env, "close")()


def main(unused_argv):
    try:
        while True:
            with sc2_env.SC2Env(map_name="MoveToBeacon",
                                players=[sc2_env.Agent(sc2_env.Race.terran)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    # default size of feature screen and feature minimap
                                    use_feature_units=True),
                                step_mul=16,  # this gives roughly 150 apm (8 would give 300 apm)
                                game_steps_per_episode=0,
                                visualize=True) as env:
                # create a keras-rl env
                keras_env = PySC2ToKerasRL_env(env)
                obs = keras_env.reset()

                # create an agent that can interact with

                # Test Agent (makes marine run in circle)
                # keras_agent = MoveToBeacon_KerasRL()
                # keras_agent.reset()
                # while True: #play the game
                #
                #     step_actions = keras_agent.step(obs)
                #     obs, reward, done, info = keras_env.step(step_actions)

                # Replace simple agent with a learning one
                # A simple model (taken from Keras-RL cartpole dqn)
                nb_actions = keras_env.action_space.n
                model = Sequential()
                model.add(Flatten(input_shape=(1,) + keras_env.observation_space.shape))
                model.add(Dense(16))
                model.add(Activation('relu'))
                model.add(Dense(16))
                model.add(Activation('relu'))
                model.add(Dense(16))
                model.add(Activation('relu'))
                model.add(Dense(nb_actions))
                model.add(Activation('linear'))
                print(model.summary())

                # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
                # even the metrics!
                memory = SequentialMemory(limit=50000, window_length=1)
                policy = BoltzmannQPolicy()
                dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                               target_model_update=1e-2, policy=policy)
                dqn.compile(Adam(lr=1e-3), metrics=['mae'])

                # Okay, now it's time to learn something!
                dqn.fit(keras_env, nb_steps=50000, visualize=False, verbose=2)


    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
