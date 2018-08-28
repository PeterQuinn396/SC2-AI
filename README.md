# SC2-AI
This is my attempt to work with Deepminds Starcraft 2 machine learning environment pysc2 to create a reinforcement learning AI.
See https://github.com/deepmind/pysc2 for all the details on the environment.

My ultimate goal would be to create an RL bot that can play Zerg (my favourite race). The rest of the community seems to be focusing more on the terran race, so this seems like a fun niche to explore.

## Table of Contents

- [A scripted Zerg bot](#a-scripted-zerg-bot) 
- [Reinforcement Learning Implementation](reinforcement-learning-implementation)
  - [Attempt #1](#attempt-1)
    - [Model Parameters](#model-parameters)
    - [Results](#results)
  - [Attempt #2](#attempt-2)

# A scripted Zerg bot

My first objective will be to get familar with the environment and then to get a simple reinforcement learning bot working in the test maps MoveToBeacon and CollectMineralShards, but first I need to learn how the PySC2 library works.

A great tutorial on the PySC2 environment can be found at https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

My implementation of it scripted_zerg_agent.py


# Reinforcement Learning Implementation 
## Attempt #1

My plan is to use the keras-rl python package to get started. I was able to write a class to wrap the standard PySC2 python environment to the API that is expected by the keras-rl learning algorithms. This also involves mapping out the action space and the observation space for the specific challenge.

I stated with the simplest challenge, MoveToBeacon. The objective is to move the single selected unit to the objective marker on the map. The observation space and action space were mapped to arrays with 4096 elements, 1 entry for each point on the 64x64 pixel map. The observation space contains a 1 where the players unit is and a 3 where the objective marker is.

A simple model for a DQN agent was copied from the keras-rl Cartpole example. The network takes in the 4096 element observation array and returns an index for the coordinates to be selected in the range of the 4096 element action space array. The environent was set to have the agent take one action every 32 game steps (~150 apm), and the training would last for 50 000 agent steps. This leads to about 830 episodes.

### Model Parameters
   
```python
 model.add(Flatten(input_shape=(1,) + keras_env.observation_space.shape))
 model.add(Dense(16))
 model.add(Activation('relu'))
 model.add(Dense(16))
 model.add(Activation('relu'))
 model.add(Dense(16))
 model.add(Activation('relu'))
 model.add(Dense(nb_actions))
 model.add(Activation('linear'))
 ```
 ### Results
Unfortunatly it didn't seem like the agent was able to learn anything. The scores it got seemed little better than random chance, and there was no improvement over time.

![results](https://github.com/PeterQuinn396/SC2-AI/blob/master/Attempt1_3x16Layers.png)

I ran another training with the layers changed to Dense(32) to see if that had any impact. It did not result in any noticable improvments.

![results](https://github.com/PeterQuinn396/SC2-AI/blob/master/Attempt1_3x32Layers.png)
 

## Attempt #2
Reduced observation space and a slimmer network
The second version of this project will involve reducing the observation space from an array of 4096 array to just 4 inputs, the x and y of the player's unit and the x,y of the objective marker. The network will also be reduced to just 2 layers. Hopefully these changes will simplify the problem and allow the agent to actually learn some good behaviours.
                
