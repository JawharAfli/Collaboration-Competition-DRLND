# Collaboration-Competition-Project- DRL NanoDegree

This project is part of Udacity's Deep Reinforcement Learning NanoDegree. Its main objective is to play a multi agent game of tennis using Unity's ML-agents tennis environment.

## Environment
![env](env.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of `+0.1`. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of `8 variables `corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. `Two continuous actions` are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of `+0.5` (over 100 consecutive episodes, after taking the maximum over both agents). Specifically:

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least `+0.5`.



## Installation

1. Clone the repository.
```
git clone https://github.com/JawharAfli/Collaboration-Competition-DRLND.git

cd Collaboration-Competition-DRLND
```

2. Prepare the environment.
```
conda env create -f environment.yml

conda activate drlnd

pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
```

3. Download the Unity ML-Agents Tennis Environment.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

4. Include the path of the tennis environment into the configuration file `config.py`.

## Train your tennis agent

```
python tennis.py --train
```
