# SC2-AI
This is my attempt to work with Deepminds Starcraft 2 machine learning environment pysc2 to create a reinforcement learning AI.
See https://github.com/deepmind/pysc2 for all the details on the environment.

My ultimate goal would be to create an RL bot that can play Zerg (my favourite race). The rest of the community seems to be focusing more on the terran race, so this seems like a fun niche to explore.

My first objective will be to get familar with the environment and then to get a simple reinforcement learning bot working in the test maps MoveToBeacon and CollectMineralShards.


## A scripted Zerg bot

A great tutorial on the PySC2 environment can be found at https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

My implementation of it scripted_zerg_agent.py


## Reinforcement Learning Implementation

My plan is to use the keras-rl python package to get started
