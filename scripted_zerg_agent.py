# a scripted zerg agent based on a tutorial by one of the contributers to the PySC2 project
# which can be found at https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import random
from absl import app


# step tells our agent what to each each time it is allowed to play
class ZergAgent(base_agent.BaseAgent):

    def unit_type_selected(self, obs, unit_type):
        if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type:
            return True

        if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type:
            return True
        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def action_available(self, obs, action):
        return action in obs.observation.available_actions

    def __init__(self):
        super(ZergAgent, self).__init__()

        self.attack_coordinates = None

    def step(self, obs):
        super(ZergAgent, self).step(obs)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative
                                  == features.PlayerRelative.SELF).nonzero()

            xmean = player_x.mean()
            ymean = player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)

        # attack if ready, or select units to attack
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) >= 20:

            if self.unit_type_selected(obs, units.Zerg.Zergling):
                if self.action_available(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now", self.attack_coordinates)

            if self.action_available(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        # see if we have spawning pool
        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)

        if len(spawning_pools) == 0:  # if not, build one if we have drones selected
            if self.unit_type_selected(obs, units.Zerg.Drone):
                if self.action_available(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)
                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))

            # if no drones selected, select drone
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)
                return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))  # select all drones

        # we have a spawning pool, so lets spawn some lings if we have already selected larva

        if self.unit_type_selected(obs, units.Zerg.Larva):

            free_supply = obs.observation.player.food_cap - obs.observation.player.food_used

            if free_supply == 0:
                if self.action_available(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")

            if self.action_available(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("now")

        # if we haven't selected larva, select it now
        larvae = self.get_units_by_type(obs, units.Zerg.Larva)

        if len(larvae) > 0:
            larva = random.choice(larvae)
            return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))

        return actions.FUNCTIONS.no_op()


# sets up the game/environment and handles looping
def main(unused_argv):
    agent = ZergAgent()
    try:
        while True:
            with sc2_env.SC2Env(map_name="AbyssalReef",
                                players=[sc2_env.Agent(sc2_env.Race.zerg),
                                         sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    use_feature_units=True),
                                step_mul=16,  # this gives roughly 150 apm (8 would give 300 apm)
                                game_steps_per_episode=0,
                                visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
