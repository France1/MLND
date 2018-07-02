## Udacity MLND Capstone Project: Learn to play Atari through Deep Reinforcement 
Learning

### Motivation
The aim of this project is to train a RL agent to play the Atari 2600 game Pong 
through two different reinforcement learning methods: DQN and Policy Gradient.
In Pong an agent and his opponent control a paddle, and the game consists of 
bouncing a ball past the other player. At each time step the agent observes a 
pixel image and chooses between two actions, Up or Down, then he receives a 
reward based on whether or not the ball went past the opponent. 

### Installation
In Anaconda create an environment called `deep-rl` as:
```
conda create --name myenv
```
then install Keras and Tensorflow which are used to compute the neural network.
The project relies on [OpenAI Gym](https://gym.openai.com/envs/Pong-v0/) to 
simulate Atari Pong. To install it run:
```
pip install gym
conda install -c anaconda cmake
pip install gym[atari]
```  

### Run
The project contains 3 python scripts:
 - `atari_random.py` simulating a random agent and is used to obtain the 
 benchmark 
 - `atari_dqn.py` for training a DQN agent
 - `atari_pg.py` for training a PG agent
which are run in a terminal window after activating the conda environment:
```
source activate deep-rl
python <script.py>
```
Each script produces a picke file which contains the results of the agent 
training. These are stores in the `results` folder

### Analysis
Preliminary analysis of the Pong environment, image pre-processing, and analysis
of the training results is described in the python notebook `Results.ipynb`. 
This produces the figures used in the report, which are saved in the 
`figures` folder 

### Report
The report of the project is contained in the `report` folder and the original
proposal submission in the `proposal` folder 


