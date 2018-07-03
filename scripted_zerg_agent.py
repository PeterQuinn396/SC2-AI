# a scripted zerg agent based on a tutorial by one of the contributers to the PySC2 project
# which can be found at https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import random
from absl import app


# step tells our agent what to each each time it is allowed to play
class ZergAgent(base_agent.BaseAgent):
    
    def step(self, obs):
        super(ZergAgent, self).step(obs)

        drones = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Zerg.Drone]

        if len(drones) > 0:
            drone = random.choice(drones)
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y)) #select all drones

        return actions.FUNCTIONS.no_op()


# sets up the game/environment and handles looping
def main (unused_argv):
    agent = ZergAgent()
    try:
        while True:
            with sc2_env.SC2Env(map_name="AbyssalReef",
                                players=[sc2_env.Agent(sc2_env.Race.zerg),
                                         sc2_env.Bot(sc2_env.Race.terran,sc2_env.Difficulty.very_easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    use_feature_units=True),
                                step_mul=16,
                                game_steps_per_episode=0,
                                visualize=True) as env:
                agent.setup(env.observation_spec(),env.action_spec())
                timesteps = env.reset()
                while True:
                    step_actions =[agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)





