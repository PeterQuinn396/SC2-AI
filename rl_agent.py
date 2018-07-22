from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

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

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY


class MoveToBeacon_KerasRL(base_agent.BaseAgent):
    action_space = [actions.FUNCTIONS.Move_screen.id, actions.FUNCTIONS.select_army.id, actions.FUNCTIONS.no_op.id]

    def __init__(self):
        super().__init__()

    def step(self, obs):
        super().step(obs)
        return actions.FUNCTIONS.no_op()


class PySC2ToKerasRL_env(rl.core.Env):
    # converts PySC2 env outputs to the inputs Keras-rl agents expect

    def __init__(self, PySC2_env):
        self.env = PySC2_env

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

        timesteps = getattr(self.env, "step")(step_actions)  # step env
        obs = timesteps[0]  # retrieve updated obs
        reward = obs.observation['score_cumulative'][0]  # get current score
        done = obs.last()  # check if its the last step
        info = None  # no debug stuff for now

        return obs, reward, done, info

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        timesteps = getattr(self.env, "reset")()
        obs = timesteps[0]  # retrieve initial obs
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



                keras_env = PySC2ToKerasRL_env(env)
                keras_agent = MoveToBeacon_KerasRL()
                keras_agent.reset()
                obs = keras_env.reset()

                # insert keras-rl leaning loop here

                while True:
                    step_actions = [keras_agent.step(obs)]
                    obs, reward, done, info = keras_env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
